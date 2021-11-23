
import numpy as np
import matplotlib.pyplot as plt
import datetime

import unittest
import tomograph
from main import back_substitution, compute_tomograph, gaussian_elimination, compute_cholesky, solve_cholesky


class Tests(unittest.TestCase):
    def test_all(self):
        print("Testing your implementations:\n")

        '''
        hope they helped you / will help you
        if you need: comment out what you don't want to test
        '''

        self.test_gaussian_elimination()
        self.test_back_substitution()
        self.test_cholesky_decomposition()
        self.test_solve_cholesky()
        self.test_compute_tomograph()

        print("\nAll tests passed!\n")

    def test_gaussian_elimination(self):
        for i in range(5):
            A = np.random.randn(4, 4)
            x = np.random.rand(4)
            b = np.dot(A, x)
            A_elim, b_elim = gaussian_elimination(A, b)
            sol = np.linalg.solve(A_elim, b_elim)
            print("test_gaussian_elimination " + str(i+1) + "/5")
            self.assertTrue(np.allclose(sol, x), "sol: " +
                            str(sol) + " x: " + str(x))
            self.assertTrue(np.allclose(A_elim, np.triu(A_elim)),
                            "A_elim: " + str(A_elim))

        A = np.array([[1., 1., 0., 0.], [0., 0., 1., 1.],
                     [0., 1., 0., 1.], [1., 0., 1., 0.]])
        x = np.array([1., 0., 0., 1.])
        b = np.dot(A, x)
        A_elim, b_elim = gaussian_elimination(A, b)
        with self.assertRaises(ValueError):
            gaussian_elimination(A, b, False)

        A = np.array([[0., 3., 5.], [3., 0., 1.], [6., 7., 2.]])
        b = np.array([23., 14., 26.])
        x = np.array([3.07619048, -0.28571429,  4.77142857])
        A_elim, b_elim = gaussian_elimination(A, b)
        sol = np.linalg.solve(A_elim, b_elim)
        self.assertTrue(np.allclose(sol, x), "sol: " +
                        str(sol) + " x: " + str(x))
        print("test_gaussian_elimination nice 6/6\n")

    def test_back_substitution(self):
        for i in range(5):
            A = np.random.randn(4, 4)
            x = np.random.rand(4)
            b = np.dot(A, x)
            A_elim, b_elim = gaussian_elimination(A, b)
            mX = back_substitution(A_elim, b_elim)
            self.assertTrue(np.allclose(mX, x), "mX: " +
                            str(mX) + " x: " + str(x))
            print("test_back_substitution " + str(i+1) + "/5")

        print("test_back_substitution nice 5/5\n")

    def test_cholesky_decomposition(self):
        for i in range(5):
            A = np.random.randn(4, 4)
            A = np.dot(A, A.T)
            L = compute_cholesky(A)
            L_0 = np.linalg.cholesky(A)
            self.assertTrue(np.allclose(L, L_0), "L: " +
                            str(L) + " L_0: " + str(L_0))
            print("test_cholesky_decomposition " + str(i+1) + "/5")
        print("test_cholesky_decomposition nice 5/5\n")

    def test_solve_cholesky(self):
        A = np.array([
            [4, 3],
            [3, 3]
        ])
        L = compute_cholesky(A)
        b = np.array([5, 2])
        x = np.linalg.solve(np.dot(L, L.T), b)
        x_0 = solve_cholesky(L, b)
        self.assertTrue(np.allclose(x, x_0), "x: " +
                        str(x) + " x_0: " + str(x_0))

        A = np.array([
            [6, 5, 2],
            [5, 8, 2],
            [2, 2, 6]])
        L = compute_cholesky(A)
        b = np.array([26, 30, 20])
        x = np.linalg.solve(np.dot(L, L.T), b)
        x_0 = solve_cholesky(L, b)
        self.assertTrue(np.allclose(x, x_0), "x: " +
                        str(x) + " x_0: " + str(x_0))
        print("test_cholesky_decomposition nice 2/2\n")

    def test_compute_tomograph(self):
        t = datetime.datetime.now()
        print("Start time: " + str(t.hour) + ":" +
              str(t.minute) + ":" + str(t.second))

        ''' 
        Compute tomographic image
        Benutzt bei der Aufgabe lieber numpy als eure Implementierung
        Die Funktion braucht lange
        '''

        n_shots = 64  # 128  # 64
        n_rays = 64  # 128  # 64
        n_grid = 32  # 64  # 32
        tim = compute_tomograph(n_shots, n_rays, n_grid)

        t = datetime.datetime.now()
        print("End time: " + str(t.hour) + ":" +
              str(t.minute) + ":" + str(t.second))

        # Visualize image
        plt.imshow(tim, cmap='gist_yarg', extent=[-1.0, 1.0, -1.0, 1.0],
                   origin='lower', interpolation='nearest')
        plt.gca().set_xticks([-1, 0, 1])
        plt.gca().set_yticks([-1, 0, 1])
        plt.gca().set_title('%dx%d' % (n_grid, n_grid))

        plt.show()

        print("test_compute_tomograph nice 1/1\n")


if __name__ == '__main__':
    unittest.main()
