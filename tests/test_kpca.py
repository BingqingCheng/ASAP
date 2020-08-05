from os import path

import ase.io
import numpy as np
from dscribe.descriptors import SOAP
from pytest import approx, raises, fail

from asaplib.reducedim import KernelPCA


def assert_array_almost_equal(first, second, tol=1e-7):
    first = np.array(first)
    second = np.array(second)
    assert first.shape == second.shape

    if np.isnan(first).any():
        fail('Not a number (NaN) found in first array')
    if np.isnan(second).any():
        fail('Not a number (NaN) found in second array')

    try:
        assert first == approx(second, abs=tol)
    except AssertionError:
        absdiff = abs(first - second)
        if np.max(absdiff) > tol:
            print('First array:\n{}'.format(first))
            print('\n \n Second array:\n{}'.format(second))
            print('\n \n Abs Difference:\n{}'.format(absdiff))
            fail('Maximum abs difference between array elements is {} at location {}'.format(np.max(absdiff),
                                                                                             np.argmax(absdiff)))


class TestKPCA(object):
    @classmethod
    def setup_class(cls):
        # todo: add docs for these
        test_folder = path.split(__file__)[0]
        fn = path.abspath(path.join(test_folder, "ice_test.xyz"))
        at_train = ase.io.read(fn, '0')
        at_test = ase.io.read(fn, '1')

        soap_desc = SOAP(species=['H', 'O'], rcut=4., nmax=6, lmax=6, sigma=0.3, periodic=True, average='off')
        desc_train = soap_desc.create(at_train)
        desc_test = soap_desc.create(at_test)

        cls.K_tr = np.dot(desc_train, desc_train.T) ** 2
        cls.K_test = np.dot(desc_test, desc_train.T) ** 2

        cls.K_tr_save = cls.K_tr.copy()
        cls.K_test_save = cls.K_test.copy()

        # references:
        cls.ref_pvec_train = np.array([[-4.06617973e+00, 1.08520804e+00, 3.69840829e-03, -1.44976066e-02],
                                       [-4.05352401e+00, 1.09320822e+00, 1.15028148e-03, 2.74168800e-02],
                                       [-3.80654705e+00, -1.10403286e+00, 1.04877050e-02, -3.63707475e-03],
                                       [-3.81287331e+00, -1.12812425e+00, 1.29987555e-02, -2.38640711e-02],
                                       [-3.80579914e+00, -1.11806303e+00, 1.15386212e-02, -2.80550703e-03],
                                       [-3.81204091e+00, -1.14223341e+00, 1.40256548e-02, -2.29474563e-02],
                                       [-4.07666395e+00, 1.09555792e+00, 4.74882612e-03, -2.29506563e-02],
                                       [-4.05571605e+00, 1.07800587e+00, 1.48251628e-03, 2.24335026e-02],
                                       [-3.79634766e+00, -1.11187590e+00, 1.04707193e-02, 8.98527376e-03],
                                       [-3.79298753e+00, -1.10617575e+00, 9.84540621e-03, 1.78169808e-02],
                                       [-4.07074480e+00, 1.11864192e+00, 1.27063488e-03, -1.80462524e-02],
                                       [-4.06688594e+00, 1.09362545e+00, 2.15794282e-03, -1.96113470e-02],
                                       [-4.05784859e+00, 1.08740112e+00, 1.47447842e-03, 1.17331366e-02],
                                       [-4.05813149e+00, 1.08739296e+00, 1.75614318e-03, 1.31801983e-02],
                                       [-3.79553551e+00, -1.11389621e+00, 1.04074380e-02, 8.63819290e-03],
                                       [-3.79218637e+00, -1.10818775e+00, 9.78369564e-03, 1.74613912e-02],
                                       [8.06281699e+00, 3.61120658e-02, 6.99331053e-01, 1.75912880e-02],
                                       [7.71411056e+00, 1.64718213e-02, -7.04066810e-01, -9.34054057e-02],
                                       [7.60371751e+00, 3.22014366e-03, -7.28918047e-01, -3.78427909e-02],
                                       [8.06310473e+00, 3.72640108e-02, 6.97723902e-01, 1.24539333e-02],
                                       [7.63289763e+00, 9.78096777e-03, -7.43053420e-01, 6.39070502e-02],
                                       [8.13279172e+00, 4.06699694e-02, 7.22168932e-01, -1.01534123e-01],
                                       [8.07660163e+00, 4.08046239e-02, 6.90906415e-01, 7.69453911e-02],
                                       [7.63397127e+00, 9.22403666e-03, -7.41389253e-01, 6.25790719e-02]],
                                      dtype="float32")

        cls.ref_pvec_test = np.array([[-4.23817197e+00, 2.21216879e-01, 1.56505888e-01, -3.56531020e-02],
                                      [-4.20017877e+00, 6.26185600e-02, 1.44533133e-01, 3.37271109e-02],
                                      [-4.28592079e+00, 2.52459297e-01, 1.68965737e-01, 2.46597830e-03],
                                      [-4.21372514e+00, 2.40352917e-01, 1.35345282e-01, -4.19430108e-02],
                                      [-4.22075792e+00, 2.57834641e-01, 1.46428658e-01, -2.54276131e-02],
                                      [-4.20912533e+00, 1.41107424e-01, 1.34349660e-01, -2.35976049e-02],
                                      [-4.23708825e+00, 8.85194416e-02, 1.61491705e-01, 1.15497682e-02],
                                      [-4.21062574e+00, 2.60190268e-01, 1.36750424e-01, 3.25269508e-03],
                                      [-3.98889410e+00, 1.56896778e-01, 7.17538880e-02, 6.39784979e-02],
                                      [-4.00525321e+00, 1.73568460e-01, 8.23704558e-02, 6.53524973e-02],
                                      [-3.65987514e+00, 6.04595020e-02, -8.57224607e-02, 1.05262634e-01],
                                      [-3.74517767e+00, -9.97008707e-01, -4.41414522e-02, -9.64487284e-02],
                                      [-4.10981478e+00, 1.66066458e-01, 1.12060385e-01, 2.31874639e-02],
                                      [-3.76780263e+00, -1.10609767e-02, -6.28042821e-02, 1.17437832e-01],
                                      [-4.04305882e+00, 2.80428956e-01, 8.85085856e-02, 9.05746488e-02],
                                      [-3.79102766e+00, -1.16690220e+00, -9.50532485e-03, -8.32897739e-02],
                                      [-4.13646883e+00, 2.12319898e-01, 1.00052036e-01, 2.64889789e-02],
                                      [-3.79810541e+00, 7.01554115e-02, -5.60348665e-02, 1.05956141e-01],
                                      [-4.05869532e+00, 3.29850364e-01, 9.29060632e-02, 9.15201392e-02],
                                      [-3.78959362e+00, -1.17952566e+00, -3.02953553e-03, -4.44245483e-02],
                                      [-4.16021423e+00, 2.67263726e-01, 1.05038377e-01, 2.16737311e-02],
                                      [-3.80467128e+00, 2.30076951e-01, -6.38924752e-02, 8.91573211e-02],
                                      [-4.03365112e+00, 3.51188385e-01, 7.69169787e-02, 7.47813405e-02],
                                      [-3.62183900e+00, -1.12477840e+00, -2.97227887e-02, 4.24700267e-02],
                                      [9.10146883e+00, 2.08341286e-01, 5.08705116e-01, -1.97210624e-01],
                                      [9.13581787e+00, 2.08796593e-01, 5.24817925e-01, -2.40741773e-01],
                                      [9.04694197e+00, 2.08949917e-01, 3.96804461e-01, -1.81276499e-01],
                                      [9.07470424e+00, 2.05175107e-01, 4.46638811e-01, -1.30784165e-01],
                                      [7.62273127e+00, -5.72024720e-02, 3.68091975e-01, -7.02586774e-02],
                                      [7.90566964e+00, 1.91416005e-02, -2.93423768e-01, -6.13016842e-02],
                                      [8.02886720e+00, 5.98404709e-03, 1.36396991e-01, -2.54598786e-01],
                                      [7.85220135e+00, -2.53487064e-02, 1.26409015e-02, -1.79153278e-01],
                                      [8.13281083e+00, 2.43322996e-02, 1.99101046e-01, -2.28687789e-01],
                                      [7.84935179e+00, -3.14337990e-02, 5.60258262e-02, -1.90401494e-01],
                                      [8.17793845e+00, 2.68832136e-02, 2.76448251e-01, -1.95965960e-01],
                                      [7.61105954e+00, -5.43027123e-02, -4.45044842e-02, -1.51450933e-01]],
                                     dtype="float32")

    def assert_kernels_not_changed(self, tol=5e-4):
        assert_array_almost_equal(self.K_tr, self.K_tr_save, tol=tol)
        assert_array_almost_equal(self.K_test, self.K_test_save, tol=tol)

    @staticmethod
    def fixcenter(kernel):
        """
        Taken from older versio of ml_kpca.py, As Is, for testing the centering even if that is removed
        """
        cernel = kernel.copy()
        cols = np.mean(cernel, axis=0)
        rows = np.mean(cernel, axis=1)
        mean = np.mean(cols)
        for i in range(len(rows)):
            cernel[i, :] -= cols
        for j in range(len(cols)):
            cernel[:, j] -= rows
        cernel += mean
        return cernel

    def test_centering(self):
        center_new = KernelPCA.center_square(self.K_tr)[2]
        center_old = self.fixcenter(self.K_tr)
        self.assert_kernels_not_changed()
        assert_array_almost_equal(center_new, center_old, tol=5e-3)

        # check the method used for the projections
        k = KernelPCA(4)
        k.fit(self.K_tr)
        # copy needed, because the class is doing that elsewhere
        # noinspection PyProtectedMember
        center_method_for_tests = k._center_test_kmat(self.K_tr.copy())
        assert_array_almost_equal(center_method_for_tests, center_old, tol=5e-3)

    def test_fit_and_transform(self):
        # init & fit
        kpca_obj = KernelPCA(4)
        kpca_obj.fit(self.K_tr)
        self.assert_kernels_not_changed()

        # assert that it raises some errors
        self.assert_local_errors(kpca_obj)

        # transform
        pvec_train = kpca_obj.transform(self.K_tr)
        self.assert_kernels_not_changed()

        # check result
        assert_array_almost_equal(pvec_train, self.ref_pvec_train, tol=5e-3)

        # transform the test
        pvec_test = kpca_obj.transform(self.K_test)
        self.assert_kernels_not_changed()
        assert_array_almost_equal(pvec_test, self.ref_pvec_test, tol=5e-3)

    def test_fit_transform(self):
        # init, fit & transform
        kpca_obj = KernelPCA(4)
        pvec_train = kpca_obj.fit_transform(self.K_tr)
        self.assert_kernels_not_changed()

        # assert that it raises some errors
        self.assert_local_errors(kpca_obj)

        # check result
        assert_array_almost_equal(pvec_train, self.ref_pvec_train, 5e-3)

        # transform the test
        pvec_test = kpca_obj.transform(self.K_test)
        self.assert_kernels_not_changed()
        assert_array_almost_equal(pvec_test, self.ref_pvec_test, tol=5e-3)

    def assert_local_errors(self, kpca_obj):
        # fixme: adthe shape and tye errors as well
        with raises(RuntimeError):
            _pvec_dummy = kpca_obj.fit(self.K_tr)
            self.assert_kernels_not_changed()
        with raises(RuntimeError):
            _pvec_dummy = kpca_obj.fit_transform(self.K_tr)
