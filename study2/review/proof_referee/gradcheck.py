import numpy as np, importlib.util, sys
src = open('/tmp/proof_referee/check_lemma_thm1.py').read().split('# ------------------------------------------------------------- Theorem 1')[1].split('def closed_form')[0]
exec("import numpy as np\n" + src.split('\n',1)[1])
rng = np.random.default_rng(3)
d,k,m = 5,2,2
J = make_J(d,k,m,0.4,0.7,0.2,"lin")
x = rng.normal(size=d*k+d*m+d*d)
f,g = J(x); e=1e-6
num = np.array([(J(x+e*np.eye(len(x))[i])[0]-J(x-e*np.eye(len(x))[i])[0])/(2*e) for i in range(len(x))])
print("max grad err", np.abs(num-g).max())
