
import numpy as np
import unittest
from main import machine_epsilon, rotation_matrix, matrix_multiplication, compare_multiplication, inverse_rotation


class Tests(unittest.TestCase):

    def test_matrix_multiplication(self):
        a = np.random.randn(2, 2)
        b = np.random.randn(2, 3)
        c = np.random.randn(3, 3)
        self.assertTrue(np.allclose(np.dot(a, a), matrix_multiplication(
            a, a)), msg="{} != {}".format(np.dot(a, a), matrix_multiplication(a, a)))
        self.assertTrue(np.allclose(np.dot(c, c), matrix_multiplication(
            c, c)), msg="{} != {}".format(np.dot(c, c), matrix_multiplication(c, c)))
        self.assertTrue(np.allclose(np.dot(a, b), matrix_multiplication(
            a, b)), msg="{} != {}".format(np.dot(a, b), matrix_multiplication(a, b)))
        self.assertTrue(np.allclose(np.dot(b, c), matrix_multiplication(
            b, c)), msg="{} != {}".format(np.dot(b, c), matrix_multiplication(b, c)))
        self.assertRaises(ValueError, matrix_multiplication, a, c)
        self.assertRaises(ValueError, matrix_multiplication, b, a)
        self.assertRaises(ValueError, matrix_multiplication, b, b)
        print("Matrix Multiplikation - Passed")

    def test_compare_multiplication(self):
        r_dict = compare_multiplication(200, 40)
        for r in zip(r_dict["results_numpy"], r_dict["results_mat_mult"]):
            self.assertTrue(np.allclose(r[0], r[1]))

    def test_machine_epsilon(self):
        eps = machine_epsilon(np.dtype(np.float32))
        self.assertTrue(np.finfo(np.float32).eps == eps,
                        msg="{} != {}".format(np.finfo(np.float32).eps, eps))

        eps = machine_epsilon(np.dtype(np.float64))
        self.assertTrue(np.finfo(np.float64).eps == eps,
                        msg="{} != {}".format(np.finfo(np.float64).eps, eps))

        print("Machine Epsilon - Passed")

    def test_rotation_matrix(self):
        theta = -90
        A = np.array([[0, 1], [-1, 0]])
        B = rotation_matrix(theta)
        self.assertTrue(np.allclose(A, B), msg="{} != {}".format(A, B))

        theta = 45
        A = np.array([[0.70710678, -0.70710678], [0.70710678, 0.70710678]])
        B = rotation_matrix(theta)

        self.assertTrue(np.allclose(A, B), msg="{} != {}".format(A, B))

        theta = 90
        A = np.array([[0, -1], [1, 0]])
        B = rotation_matrix(theta)
        self.assertTrue(np.allclose(A, B), msg="{} != {}".format(A, B))

        theta = 30
        A = np.array([[0.8660254, -0.5], [0.5, 0.8660254]])
        B = rotation_matrix(theta)
        self.assertTrue(np.allclose(A, B), msg="{} != {}".format(A, B))

        print("Rotation Matrix - Passed")

    def test_inverse_rotation(self):
        theta = -90
        A = np.array([[0, -1], [1, 0]])
        B = inverse_rotation(theta)
        self.assertTrue(np.allclose(A, B), msg="{} != {}".format(A, B))

        theta = 45
        A = np.array([[0.70710678, 0.70710678], [-0.70710678, 0.70710678]])
        B = inverse_rotation(theta)
        self.assertTrue(np.allclose(A, B), msg="{} != {}".format(A, B))

        theta = 90
        A = np.array([[0, 1], [-1, 0]])
        B = inverse_rotation(theta)
        self.assertTrue(np.allclose(A, B), msg="{} != {}".format(A, B))

        theta = 30
        A = np.array([[0.8660254, 0.5], [-0.5, 0.8660254]])
        B = inverse_rotation(theta)
        self.assertTrue(np.allclose(A, B), msg="{} != {}".format(A, B))

        print("Inverse Rotation Matrix - Passed")

    def test_all(self):
        self.test_matrix_multiplication()
        self.test_compare_multiplication()
        self.test_machine_epsilon()
        self.test_rotation_matrix()
        self.test_inverse_rotation()

        print("\nAll Tests - Passed")


if __name__ == '__main__':
    unittest.main()
