import pytest
import numpy as np
import GMRES as GM

from GMRES import gmres

# Empty test file for later tests

def test_example():
    A = np.array([[1,4,7],[2,9,7],[5,8,3]])
    b = np.array([1,8,2]).reshape(3,1)
    alg = GM.gmres()
    x0 = np.zeros((3,1))
    x = alg.mygmres(3,b, x0,3,np.eye(3),A)

    xactual = np.array([-2.18,1.84,-0.6]).reshape(3,1)
    assert np.allclose(x,xactual,atol=1e-2)

def test_example2():
    A = np.array([[1,4,7],[2,9,7],[5,8,3]])
    b = np.array([[1,2,5],[8,3,-3],[2,9,8]])
    b1 = b[:,1].reshape(3,1)
    b2 = b[:,2].reshape(3,1)
    alg = GM.gmres()
    x0 = np.zeros((3,1))
    x1 = alg.mygmres(3,b1, x0,3,np.eye(3),A)

    x1actual = np.array([2.1,-.22,.11]).reshape(3,1)
    assert np.allclose(x1,x1actual,atol=1e-1)

    x2 = alg.mygmres(3,b2, x0,3,np.eye(3),A)
    x2actual = np.array([4.8,-2.6,1.5]).reshape(3,1)
    assert np.allclose(x2,x2actual,atol=1e-1)
