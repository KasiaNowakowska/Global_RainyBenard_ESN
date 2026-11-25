"""
Dedalus script for Rainy Benard study using dedalus v3.

Usage: rainy_v3.py [--Rayleigh=<Rayleigh> --beta=<beta> --gamma=<gamma> --alpha=<alpha> --nx=<nx> --nz=<nz> --Lx=<Lx> --RH0=<RH0> --RHb=<RHb> --RHt=<RHt> --simtime=<simtime> --max_dt=<max_dt> --restartfile=<file> --restart]

Options:
    --Rayleigh=<Rayleigh>    Rayleigh number [default: 1e6]
    --alpha=<alpha>          Clausius Clapeyron parameter [default: 3.0]
    --beta=<beta>            beta parameter [default: 1.201]
    --gamma=<gamma>          condensational heating parameter [default: 0.5]
    --nx=<nx>                x (Fourier) resolution [default: 256]
    --nz=<nz>                vertical z (Chebyshev) resolution [default: 64]
    --Lx=<Lx>                x length [default: 20]
    --RH0=<RH0>              the percentage of humidity in the interior for BG IC [default: 0.75]
    --RHb=<RHb>              the percentage humidity on the unsaturated lower boundary [default: 0.75]
    --RHt=<RHt>              the percentage humidity on the upper boundary RHt=1 saturated [default: 1]
    --simtime=<simtime>      simulation run time [default: 1200]
    --max_dt=<max_dt>        maximum timestep [default: 1e-2]
    --restartfile=<file>     file to restart simulation from 
    --restart                apply this to cause restart from file
"""

#import packages
import logging
logger = logging.getLogger(__name__)
for system in ['h5py._conv', 'matplotlib', 'PIL']:
     logging.getLogger(system).setLevel(logging.WARNING)

import numpy as np
import h5py
import sys

from docopt import docopt
args = docopt(__doc__)

print(sys.argv)
restart = (len(sys.argv) > 1 and sys.argv[-1] == '--restart') #allows restart from command line need to check if better way to do it
print(restart)

from mpi4py import MPI
nproc = MPI.COMM_WORLD.size

#Parameters
Lx = float(args['--Lx'])
Lz = 1
nx = float(args['--nx'])
nz = float(args['--nz'])

beta = float(args['--beta'])
Rayleigh = float(args['--Rayleigh'])
alpha = float(args['--alpha'])
gamma = float(args['--gamma'])
Prandtl = 1
Prandtlm = 1
Sm = 1
T1 = 0
RH0 = float(args['--RH0'])
RHb = float(args['--RHb'])
RHt = float(args['--RHt'])
simtime = float(args['--simtime'])
restart_file_path = args['--restartfile']

data_dir = 'scratch/edited_new_file_Ra{0:.2e}_RH0{1:}_RHb{2:}_RHt{3:}_gamma{4:}_nx{5:}_nz{6:}_Lx{7:}_simtime{8:}'.format(float(args['--Rayleigh']), args['--RH0'], args['--RHb'], args['--RHt'], args['--gamma'], args['--nx'], args['--nz'], args['--Lx'], args['--simtime'])

import dedalus.tools.logging as dedalus_logging
dedalus_logging.add_file_handler(data_dir+'/logs/dedalus_log', 'DEBUG')

import dedalus.public as de
from dedalus.extras import flow_tools


logger.info('α={:}, β={:}, γ={:}, Ra={:.2e}'.format(alpha,beta,gamma, Rayleigh))

run_time = 1200
run_time_iter = 200

P = (Rayleigh * Prandtl)**(-1/2)
S = (Rayleigh * Prandtlm)**(-1/2)
PdR = (Prandtl / Rayleigh)**(1/2)
PtR = 1
tau = 2e-1
k = 1e5
max_dt = float(args['--max_dt'])

dealias = 3/2
dtype = np.float64

coords = de.CartesianCoordinates('y', 'x', 'z', right_handed=False)
#coords = de.CartesianCoordinates('x', 'y', 'z')
dist = de.Distributor(coords, mesh=[1,nproc], dtype=dtype)

xb = de.RealFourier(coords['x'], size=nx, bounds=(0, Lx), dealias=dealias)
zb = de.ChebyshevT(coords['z'], size=nz, bounds=(0, Lz), dealias=dealias)
x = dist.local_grid(xb)
z = dist.local_grid(zb)

bases = (xb, zb)

p = dist.Field(name='p', bases=bases)
u = dist.VectorField(coords, name='u', bases=bases)
b = dist.Field(name='b', bases=bases)
q = dist.Field(name='q', bases=bases)

tau_p = dist.Field(name='tau_p')
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xb)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xb)
tau_b1 = dist.Field(name='tau_b1', bases=xb)
tau_b2 = dist.Field(name='tau_b2', bases=xb)
tau_q1 = dist.Field(name='tau_q1', bases=xb)
tau_q2 = dist.Field(name='tau_q2', bases=xb)

''' new lift bases
lift_basis = zb.derivative_basis(1)
lift = lambda A: de.Lift(A, lift_basis, -1)
lift_basis2 = zb.derivative_basis(2)
lift2 = lambda A: de.Lift(A, lift_basis2, -1)
lift2_2 = lambda A: de.Lift(A, lift_basis2, -2)
'''

# old lift bases
zb1 = zb.clone_with(a=zb.a+1, b=zb.b+1)
zb2 = zb.clone_with(a=zb.a+2, b=zb.b+2)
lift1 = lambda A, n: de.Lift(A, zb1, n)
lift = lambda A, n: de.Lift(A, zb2, n)

ey, ex, ez = coords.unit_vector_fields(dist)

H = lambda A: 0.5*(1+np.tanh(k*A))

z_grid = dist.Field(name='z_grid', bases=zb)
z_grid['g'] = z

T = b - beta*z_grid
qs = np.exp(alpha*T)
rh = q/qs

curl = lambda A: de.Curl(A)
grad = lambda A: de.Gradient(A, coords)
trans = lambda A: de.TransposeComponents(A)
cross = lambda A,B: de.CrossProduct(A, B)

omega = curl(u)


''' IVP problem with new bases
vars = [p, u, b, q, tau_p, tau_u1, tau_u2, tau_b1, tau_b2, tau_q1, tau_q2]
problem = de.IVP(vars, namespace=locals())

problem.add_equation('div(u) + lift(tau_p) = 0')
problem.add_equation('dt(b) - P*lap(b) + lift2_2(tau_b1) + lift2(tau_b2) = - (u@grad(b)) + gamma/tau*((q-qs)*H(q-qs))')
problem.add_equation('dt(u) - PdR*lap(u) + grad(p) - PtR*b*ez + lift2_2(tau_u1) + lift2(tau_u2) = cross(u, omega)')
problem.add_equation('dt(q) - S*lap(q) + lift2_2(tau_q1) + lift2(tau_q2) = - (u@grad(q)) - 1/tau*((q-qs)*H(q-qs))')
problem.add_equation('b(z=0) = T1')
problem.add_equation('b(z=Lz) = beta + T1 - 1') # technically β*Lz
problem.add_equation('q(z=0) = RHb')
#problem.add_equation('q(z=Lz) = np.exp(alpha*(T1-1))') #saturated upper boundary
problem.add_equation('q(z=Lz) = RHt*np.exp(alpha*(T1-1))') #unsaturated upper boundary - matches IC
problem.add_equation('u(z=0) = 0')
problem.add_equation('u(z=Lz) = 0')
problem.add_equation('integ(p) = 0')
#problem.add_equation('dt(q) = dqdt')
'''


#IVP problem with old bases
vars = [p, u, b, q, tau_p, tau_u1, tau_u2, tau_b1, tau_b2, tau_q1, tau_q2]
problem = de.IVP(vars, namespace=locals())

problem.add_equation('div(u) + tau_p + 1/PdR*dot(lift(tau_u2,-1),ez) = 0')
problem.add_equation('dt(b) - P*lap(b) + lift(tau_b1, -1) + lift(tau_b2, -2) = - (u@grad(b)) + gamma/tau*((q-qs)*H(q-qs))')
problem.add_equation('dt(u) - PdR*lap(u) + grad(p) - PtR*b*ez + lift(tau_u1, -1) + lift(tau_u2, -2) = cross(u, omega)')
problem.add_equation('dt(q) - S*lap(q) + lift(tau_q1, -1) + lift(tau_q2, -2) = - (u@grad(q)) - 1/tau*((q-qs)*H(q-qs))')
problem.add_equation('b(z=0) = T1')
problem.add_equation('b(z=Lz) = beta + T1 - 1') # technically β*Lz
problem.add_equation('q(z=0) = RHb')
#problem.add_equation('q(z=Lz) = np.exp(alpha*(T1-1))') #saturated upper boundary
problem.add_equation('q(z=Lz) = RHt*np.exp(alpha*(T1-1))') #unsaturated upper boundary - matches IC
problem.add_equation('u(z=0) = 0')
problem.add_equation('u(z=Lz) = 0')
problem.add_equation('integ(p) = 0')

#build solver
ts = de.SBDF2
cfl_safety_factor = 0.2

solver = problem.build_solver(ts)
solver.stop_sim_time = simtime + 20*restart
solver.stop_iteration = np.inf


#Initial Conditions
if not restart:
    amp = 1e-1
    
    b.fill_random('g', seed=42, distribution='normal', scale=amp) # Random noise
    #b['g'] *= z * (Lz - 1)
    b['g'] += (beta - 1) * z
    
    q.fill_random('g', seed=42, distribution='normal', scale=amp) # Random noise
    q['g'] += RH0 * np.exp(alpha * (b['g'] - beta * z))
    
    #file_handler_mode = 'overwrite'
    dt = max_dt = min(max_dt, tau/4)
    logger.info('setting dt = min(max_dt={:.2g}, tau/4={:.2g})'.format(max_dt, tau/4))
else:
    restart_file = restart_file_path
    write, dt = solver.load_state(restart_file)
    #file_handler_mode = 'append'
    dt = 1e-2
    max_dt = min(max_dt, tau/4)
    logger.info('setting dt = {:.2g}'.format(max_dt))


cfl = flow_tools.CFL(solver, dt, safety=cfl_safety_factor, cadence=1, threshold=0.1, max_change=1.5, min_change=0.5, max_dt=max_dt)
cfl.add_velocity(u)

report_cadence = 10

KE = 0.5*(u@u)
Re = np.sqrt(u@u)/PdR

vol = Lx*Lz
integ = lambda A: de.Integrate(de.Integrate(A, 'x'), 'z')
vol_avg = lambda A: integ(A)/vol

# Global volume-averaged observables used by the ESN input data.
timeseries = solver.evaluator.add_file_handler(data_dir+'/timeseries', sim_dt=1, parallel="gather")
timeseries.add_task(vol_avg(KE), name='KE')
timeseries.add_task(vol_avg(q), name='moisture')

# Save the full solver state so the simulation can be restarted.
checkpoints = solver.evaluator.add_file_handler(data_dir+'/checkpoints', sim_dt=100, max_writes=1, parallel="gather")
checkpoints.add_tasks(solver.state)

flow = flow_tools.GlobalFlowProperty(solver, cadence=report_cadence)
flow.add_property(Re, name='Re')
flow.add_property(KE, name='KE')


good_solution = True
KE_avg = 0
try:
    while solver.proceed and good_solution:
        # advance
        solver.step(dt)
        if solver.iteration % report_cadence == 0:
            Re_max = flow.max('Re')
            Re_avg = flow.volume_integral('Re')/Lx/Lz
            KE_avg = flow.volume_integral('KE')/Lx/Lz
            log_string = 'Iteration: {:5d}, Time: {:8.3e}, dt: {:5.1e}'.format(solver.iteration, solver.sim_time, dt)
            log_string += ', KE: {:.2g}, Re: {:.2g} ({:.2g})'.format(KE_avg, Re_avg, Re_max)
            logger.info(log_string)
        dt = cfl.compute_timestep()
        good_solution = np.isfinite(dt)*np.isfinite(KE_avg)
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
