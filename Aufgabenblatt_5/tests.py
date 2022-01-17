import os
import numpy as np
import unittest
import time
import matplotlib.pyplot as plt


from lib import idft, dft, ifft, plot_harmonics, read_audio_data, write_audio_data
from main import (
    dft_matrix,
    is_unitary,
    create_harmonics,
    shuffle_bit_reversed_order,
    fft,
    generate_tone,
    low_pass_filter,
)


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.path.isfile("data.npz"):
            cls.data = np.load("data.npz", allow_pickle=True)
        else:
            raise IOError("Could not load data file 'data.npz' for tests.")

    @classmethod
    def tearDownClass(cls):
        cls.data.close()

    def test_all(self):
        print("\nTest 1:")
        self.test_1_dft_matrix()
        print("\nTest 2:")
        self.test_2_is_unitary()
        print("\nTest 3:")
        self.test_3_create_harmonics()
        print("\nTest 4:")
        self.test_4_shuffle_bit_reversed_order()
        print("\nTest 5:")
        self.test_5_fft()

    def test_1_dft_matrix(self):
        dftSimple = dft_matrix(2)
        dftSimpleNp = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        print("\nDEINE DFT:")
        print(dftSimple)
        print("\nNUMPY DFT:")
        print(dftSimpleNp)
        print("")
        self.assertTrue(np.allclose(dftSimple, dftSimpleNp))

        dft1 = dft_matrix(n=16)
        self.assertTrue(dft1.shape[0] == dft1.shape[1] == 16)
        self.assertTrue(np.allclose(dft1, Tests.data["t1_dft1"]))

        dft2 = dft_matrix(n=64)
        self.assertTrue(dft2.shape[0] == dft2.shape[1] == 64)
        self.assertTrue(np.allclose(dft2, Tests.data["t1_dft2"]))

        signal = np.random.rand(64)
        self.assertTrue(np.allclose(idft(dft(signal)), signal))
        self.assertTrue(
            np.allclose(np.fft.fft(signal) / np.sqrt(signal.size), dft(signal))
        )
        print("OK!")

    #        np.savez("data", t1_dft1=dft1, t1_dft2=dft2)

    def test_2_is_unitary(self):
        self.assertFalse(is_unitary(Tests.data["t2_m1"]))

        signal = np.random.rand(64)
        self.assertTrue(np.allclose(idft(dft(signal, True)), signal))
        self.assertTrue(is_unitary(Tests.data["t2_m2"]))

        m1 = np.random.rand(16, 16)
        m2 = dft_matrix(32)
        print("OK!")

    #        np.savez("data1", t2_m1=m1, t2_m2=m2)

    def test_3_create_harmonics(self):
        s1, fs1 = create_harmonics(16)
        self.assertTrue(len(s1) == len(fs1) == 16)
        self.assertTrue(np.allclose(s1, Tests.data["t3_s1"]))
        self.assertTrue(np.allclose(fs1, Tests.data["t3_fs1"]))

        s2, fs2 = create_harmonics()
        self.assertTrue(len(s2) == len(fs2) == 128)
        self.assertTrue(np.allclose(s2, Tests.data["t3_s2"]))
        self.assertTrue(np.allclose(fs2, Tests.data["t3_fs2"]))

        # plot_harmonics(s2, fs2)
        print("OK!")

    #        np.savez("data3", t3_s1=s1, t3_fs1=fs1, t3_s2=s2, t3_fs2=fs2)

    def test_4_shuffle_bit_reversed_order(self):
        d1 = shuffle_bit_reversed_order(np.linspace(0, 15, 16))
        d2 = shuffle_bit_reversed_order(np.linspace(0, 8191, 8192))
        self.assertTrue(np.allclose(d1, Tests.data["t4_d1"]))
        self.assertTrue(np.allclose(d2, Tests.data["t4_d2"]))
        print("OK!")

    #        np.savez("data4", t4_d1=d1, t4_d2=d2)

    def test_5_fft(self):
        # vom Arbeitsblatt ganz unten das Beispiel
        data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.complex128)
        data1 = ifft(fft(data))
        dataEnd = fft(data)
        data2 = np.fft.fft(data) / np.sqrt(data.size)
        print("\nDEIN ARRAY AM ENDE")
        print(dataEnd)  # DEINS
        print("\nDIE NUMPY LÖSUNG:")
        print(data2)  # LÖSUNG
        print("")
        self.assertTrue(np.allclose(data, data1))
        self.assertTrue(np.allclose(fft(data), np.fft.fft(data) / np.sqrt(data.size)))

        data = np.random.randn(128)
        data1 = ifft(fft(data))
        self.assertTrue(np.allclose(data, data1))
        self.assertTrue(np.allclose(fft(data), np.fft.fft(data) / np.sqrt(data.size)))
        print("OK!")


if __name__ == "__main__":
    unittest.main()
