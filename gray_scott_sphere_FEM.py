# gray_scott_sphere_FEM.py
# Gray–Scott on a sphere surface (Laplace–Beltrami via surface FEM, DOLFINx)

# imports
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem
from dolfinx.io import XDMFFile, gmshio
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
import gmsh
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Parallel processing
comm = MPI.COMM_WORLD   # group containing all parallel processes
rank = comm.rank        # ID of this process (0 = master)

# Parameters 
Du, Dv = 2e-4, 1e-4       # diffusion 
F, k   = 0.029, 0.027     # other popular values: (0.026,0.051), (0.020,0.050)
dt = 0.25                 # smaller dt helps explicit reactions
num_steps = 10000
output_every = 250


# Sphere surface mesh (tdim=2, gdim=3)
R = 1.0 # Unit sphere
h = 0.06  # start moderate; refine later (error reduces with mesh size)

# descriptive run tag + output paths 
def build_run_tag():
    # Pull what exists from globals and skip any missing
    parts = []
    for key, fmt in [
        ("F",       "F{:.3f}"),
        ("k",       "k{:.3f}"),
        ("Du",      "Du{:.1e}"),
        ("Dv",      "Dv{:.1e}"),
        ("dt",      "dt{:.2f}"),
        ("h",       "h{:.2f}"),
        ("num_steps","steps{}"),   
    ]:
        if key in globals() and globals()[key] is not None:
            val = globals()[key]
            parts.append(fmt.format(val))
    return "_".join(parts) if parts else "run"

run_tag = build_run_tag()

plot_dir = Path("gs_plots") / run_tag
plot_dir.mkdir(parents=True, exist_ok=True)

# Useful filenames for later (static images if you want them)
png_u = plot_dir / f"u_{run_tag}.png"
png_v = plot_dir / f"v_{run_tag}.png"
png_uv = plot_dir / f"uv_{run_tag}.png"

# build mesh 
gmsh.initialize()
gmsh.model.add("sphere_surface")
_ = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, R) # produces a 3D solid sphere, not just the surface yet, _ = ignores the returned shape ID
gmsh.model.occ.synchronize() 
surfs = gmsh.model.getEntities(dim=2) # extract 2D surfaces in the model
gmsh.model.addPhysicalGroup(2, [s[1] for s in surfs], tag=1) # maps the sphere’s surface into a named group so that DOLFINx knows this is the domain
gmsh.model.setPhysicalName(2, 1, "SphereSurface") 
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h) # Use mesh elements of size ~ h everywhere on the surface 
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h) # min = max will force a uniform mesh size, giving you a quasi-uniform triangulation of the sphere.
gmsh.model.mesh.generate(2) # manifold mesh (dim=2, embedding dim=3). 
domain, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh.model, comm, rank, gdim=3) # converts Gmsh’s mesh → DOLFINx mesh
gmsh.finalize()


# Function space and fields (using FEniCSx syntax)
V = fem.functionspace(domain, ("Lagrange", 1))   # P1 scalar FEM space
# Current fields (at time t)
u = fem.Function(V, name="u")    # chemical species U
v = fem.Function(V, name="v")    # chemical species V
# Previous time-step fields (at time t_n)
u_n = fem.Function(V, name="u")  # U field at previous timestep
v_n = fem.Function(V, name="v")  # V field at previous timestep


# Initial condition: positive v everywhere + many random caps
def make_multi_cap_ic(K=24, theta0=0.30,         # ~17° caps
                      u_out=0.98, v_out=0.05,    # <-- v baseline > 0
                      u_in=0.42,  v_in=0.38,     # finite-amplitude kick
                      seed=7, noise=0.01):
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(K, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    cos0 = np.cos(theta0)

    def do_field(X, base_out, base_in):
        x, y, z = X[0], X[1], X[2]
        Rn = np.sqrt(x*x + y*y + z*z)
        nx, ny, nz = x / Rn, y / Rn, z / Rn
        dots = (dirs[:, 0, None] * nx +
                dirs[:, 1, None] * ny +
                dirs[:, 2, None] * nz)              # (K, npts)
        mask = np.max(dots, axis=0) > cos0
        out = np.full_like(x, base_out)
        out[mask] = base_in
        out += noise * rng.standard_normal(out.shape) # tiny global noise
        return np.clip(out, 0.0, 1.0)

    def u_ic(X): return do_field(X, u_out, u_in)
    def v_ic(X): return do_field(X, v_out, v_in)
    return u_ic, v_ic

u_ic, v_ic = make_multi_cap_ic()
u_n.interpolate(u_ic); v_n.interpolate(v_ic)
u.interpolate(u_ic);   v.interpolate(v_ic)


# frame saver for movie (optional)
tdim = domain.topology.dim
domain.topology.create_connectivity(tdim, 0)   # cells -> vertices
cells = domain.topology.connectivity(tdim, 0).array.reshape(-1, 3)
coords = domain.geometry.x
faces_xyz = coords[cells]  # geometry is fixed in time

def save_u_frame(u, step, t):
    """Save a PNG of u on the sphere for this step."""
    if rank != 0:
        return  # only rank 0 plots in parallel runs

    u_vals = u.x.array
    tri_vals = u_vals[cells].mean(axis=1)

    # quantile-based normalization for better contrast
    lo, hi = np.quantile(tri_vals, [0.02, 0.98])
    if hi <= lo:
        lo, hi = float(tri_vals.min()), float(tri_vals.max() + 1e-12)

    norm = mpl.colors.Normalize(vmin=float(lo), vmax=float(hi))
    cmap = plt.get_cmap("viridis")
    face_colors = cmap(norm(tri_vals))

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    coll = Poly3DCollection(faces_xyz, facecolors=face_colors, edgecolors="none")
    ax.add_collection3d(coll)
    ax.auto_scale_xyz(coords[:, 0], coords[:, 1], coords[:, 2])
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()

    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cb = plt.colorbar(mappable, ax=ax, shrink=0.7, pad=0.05)
    cb.set_label("u")

    ax.set_title(f"Gray–Scott on sphere — u\nstep={step}, t={t:.2f}")
    plt.tight_layout()

    # numbered filename for movie building
    frame_path = plot_dir / f"u_frame_{step:05d}.png"
    plt.savefig(frame_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved frame {frame_path}")



# Weak forms (Laplace–Beltrami comes for free on the surface mesh)
phi = ufl.TestFunction(V) # test function
u_trial = ufl.TrialFunction(V)
v_trial = ufl.TrialFunction(V)
dx = ufl.dx
grad = ufl.grad

# Weak form 
a_u = (u_trial*phi + dt*Du*ufl.inner(grad(u_trial), grad(phi))) * dx
a_v = (v_trial*phi + dt*Dv*ufl.inner(grad(v_trial), grad(phi))) * dx

def rhs_u_form(u_old, v_old):
    R_u = -u_old*v_old*v_old + F*(1.0 - u_old)
    return (u_old*phi + dt*R_u*phi) * dx

def rhs_v_form(u_old, v_old):
    R_v =  u_old*v_old*v_old - (F + k)*v_old
    return (v_old*phi + dt*R_v*phi) * dx

A_u = assemble_matrix(fem.form(a_u)); A_u.assemble()
A_v = assemble_matrix(fem.form(a_v)); A_v.assemble()

def make_solver(A):
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.CG)
    ksp.getPC().setType(PETSc.PC.Type.GAMG)
    ksp.setTolerances(rtol=1e-8, atol=1e-12, max_it=400)
    return ksp

ksp_u = make_solver(A_u)
ksp_v = make_solver(A_v)

L_u_form = fem.form(rhs_u_form(u_n, v_n))
L_v_form = fem.form(rhs_v_form(u_n, v_n))
b_u = create_vector(L_u_form)
b_v = create_vector(L_v_form)

# -----------------------
# Time loop (+ positivity clamp)
# -----------------------
with XDMFFile(comm, "gray_scott_sphere.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u_n, 0.0)
    xdmf.write_function(v_n, 0.0)

    t = 0.0
    for step in range(1, num_steps + 1):
        t += dt

        L_u_form = fem.form(rhs_u_form(u_n, v_n))
        L_v_form = fem.form(rhs_v_form(u_n, v_n))
        b_u.set(0.0); b_v.set(0.0)
        assemble_vector(b_u, L_u_form)
        assemble_vector(b_v, L_v_form)
        b_u.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        b_v.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        ksp_u.solve(b_u, u.x.petsc_vec)
        ksp_v.solve(b_v, v.x.petsc_vec)
        u.x.scatter_forward(); v.x.scatter_forward()

        # --- positivity clamp to avoid v dipping negative then dying ---
        np.clip(u.x.array, 0.0, 1.2, out=u.x.array)
        np.clip(v.x.array, 0.0, 1.2, out=v.x.array)

        u_n.x.array[:] = u.x.array
        v_n.x.array[:] = v.x.array

        if (step % output_every == 0) or (step == num_steps):
            xdmf.write_function(u, t)
            xdmf.write_function(v, t)
            if rank == 0:
                umin, umax = float(u.x.array.min()), float(u.x.array.max())
                vmin, vmax = float(v.x.array.min()), float(v.x.array.max())
                print(f"t={t:.1f}, step {step}/{num_steps} "
                      f"| u[{umin:.3f},{umax:.3f}] v[{vmin:.3f},{vmax:.3f}]")

                # save movie frame of u
                save_u_frame(u, step, t)
