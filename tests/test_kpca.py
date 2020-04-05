import ase.io
import numpy as np
from dscribe.descriptors import SOAP
from pytest import approx, raises, fail

from asaplib.pca import KernelPCA


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
        fn = "ice_test.xyz"
        at_train = ase.io.read(fn, '0')
        at_test = ase.io.read(fn, '1')

        soap_desc = SOAP(species=['H', 'O'], rcut=4., nmax=6, lmax=6, sigma=0.3, periodic=True)
        desc_train = soap_desc.create(at_train)
        desc_test = soap_desc.create(at_test)

        cls.K_tr = np.dot(desc_train, desc_train.T) ** 2
        cls.K_test = np.dot(desc_test, desc_train.T) ** 2

        cls.K_tr_save = cls.K_tr.copy()
        cls.K_test_save = cls.K_test.copy()

        # references:
        cls.ref_pvec_train = np.array([[-3.22200775e+00, 9.59025264e-01, 2.57486105e-03, -1.31988525e-02],
                                       [-3.21541214e+00, 9.64409232e-01, 1.82008743e-03, 3.45230103e-02],
                                       [-2.88484693e+00, -9.90488589e-01, 7.56590813e-03, -7.28607178e-03],
                                       [-2.88778687e+00, -1.01122987e+00, 9.33682173e-03, -2.76336670e-02],
                                       [-2.88088942e+00, -1.00375652e+00, 8.11238587e-03, -2.47955322e-03],
                                       [-2.88371611e+00, -1.02455544e+00, 9.85565037e-03, -2.25982666e-02],
                                       [-3.23374653e+00, 9.69168663e-01, 2.98285484e-03, -2.54135132e-02],
                                       [-3.21809673e+00, 9.51697350e-01, 1.98252499e-03, 2.71377563e-02],
                                       [-2.87221289e+00, -9.98565197e-01, 7.82093406e-03, 9.64355469e-03],
                                       [-2.87068844e+00, -9.94239807e-01, 7.89511204e-03, 2.07443237e-02],
                                       [-3.23205018e+00, 9.90474820e-01, 7.16954470e-04, -2.46047974e-02],
                                       [-3.22614098e+00, 9.67900276e-01, 1.49932504e-03, -2.57110596e-02],
                                       [-3.21842861e+00, 9.60623264e-01, 1.48800015e-03, 1.26113892e-02],
                                       [-3.21877217e+00, 9.60831106e-01, 1.73293054e-03, 1.49383545e-02],
                                       [-2.87132740e+00, -1.00060081e+00, 7.80840218e-03, 9.06372070e-03],
                                       [-2.86981440e+00, -9.96274173e-01, 7.88625330e-03, 2.02102661e-02],
                                       [6.23684359e+00, 4.59359102e-02, 5.90664506e-01, 1.37786865e-02],
                                       [5.99937296e+00, 3.21112685e-02, -5.93227923e-01, -6.92749023e-02],
                                       [5.92144489e+00, 1.98389031e-02, -6.14026666e-01, -2.38800049e-02],
                                       [6.23685551e+00, 4.71412316e-02, 5.89428842e-01, 1.00402832e-02],
                                       [5.93770933e+00, 2.54049711e-02, -6.24091208e-01, 4.53033447e-02],
                                       [6.29011536e+00, 5.04831225e-02, 6.08533144e-01, -7.43865967e-02],
                                       [6.24502945e+00, 4.98473085e-02, 5.84447801e-01, 5.39855957e-02],
                                       [5.93856430e+00, 2.47935243e-02, -6.22833610e-01, 4.44641113e-02]],
                                      dtype="float32")

        cls.ref_pvec_test = np.array([[-3.35908103e+00, 1.41343698e-01, 1.47953272e-01, 3.27377319e-02],
                                      [-3.31612492e+00, -1.07574072e-02, 1.35046065e-01, 1.33384705e-01],
                                      [-3.41174889e+00, 1.67209089e-01, 1.58263341e-01, 9.24377441e-02],
                                      [-3.32852888e+00, 1.55460045e-01, 1.25684977e-01, 2.61993408e-02],
                                      [-3.34114552e+00, 1.61819428e-01, 1.38998210e-01, 4.04891968e-02],
                                      [-3.32662868e+00, 6.52360991e-02, 1.24293432e-01, 5.07812500e-02],
                                      [-3.35368443e+00, 1.46444701e-02, 1.51635304e-01, 1.10145569e-01],
                                      [-3.33579969e+00, 1.77854657e-01, 1.28142744e-01, 6.83593750e-02],
                                      [-3.09308386e+00, 9.19110030e-02, 7.09910020e-02, 1.36192322e-01],
                                      [-3.11787128e+00, 1.06056929e-01, 8.38626921e-02, 1.32904053e-01],
                                      [-2.73938131e+00, -7.63382576e-03, -7.46425763e-02, 1.47666931e-01],
                                      [-2.81615782e+00, -8.74039114e-01, -5.04961833e-02, -1.79351807e-01],
                                      [-3.22462940e+00, 8.83198604e-02, 1.08695842e-01, 8.92791748e-02],
                                      [-2.85119152e+00, -6.01680577e-02, -6.00574091e-02, 1.95388794e-01],
                                      [-3.15944910e+00, 1.97961837e-01, 8.99276808e-02, 1.79084778e-01],
                                      [-2.86026454e+00, -1.02728796e+00, -1.66856572e-02, -1.31172180e-01],
                                      [-3.25423741e+00, 1.28395483e-01, 9.27777514e-02, 9.88464355e-02],
                                      [-2.88313174e+00, 1.34271830e-02, -5.44521436e-02, 1.85310364e-01],
                                      [-3.17692256e+00, 2.42637068e-01, 9.33682919e-02, 1.80061340e-01],
                                      [-2.86172438e+00, -1.04268909e+00, -7.91028887e-03, -7.27233887e-02],
                                      [-3.28557396e+00, 1.81960434e-01, 9.90050808e-02, 9.14001465e-02],
                                      [-2.89285851e+00, 1.53065950e-01, -6.27451614e-02, 1.49971008e-01],
                                      [-3.15397811e+00, 2.64356166e-01, 7.86151141e-02, 1.46804810e-01],
                                      [-2.67891097e+00, -1.00727737e+00, -2.11720690e-02, 4.85687256e-02],
                                      [6.93251991e+00, 1.99054211e-01, 4.15148318e-01, -1.47033691e-01],
                                      [6.97309780e+00, 2.04649135e-01, 4.33929205e-01, -1.86050415e-01],
                                      [6.90536976e+00, 2.03557119e-01, 3.23883027e-01, -1.38900757e-01],
                                      [6.91267300e+00, 1.95674390e-01, 3.64136517e-01, -1.01211548e-01],
                                      [5.94695473e+00, -4.40106392e-02, 3.08899760e-01, -4.16564941e-02],
                                      [6.14393854e+00, 3.34385335e-02, -2.48942226e-01, -5.07354736e-02],
                                      [6.23361588e+00, 2.09025517e-02, 1.08646035e-01, -1.84219360e-01],
                                      [6.09808159e+00, -1.34187452e-02, 4.43854928e-03, -1.26083374e-01],
                                      [6.30923796e+00, 4.06269431e-02, 1.65124312e-01, -1.70028687e-01],
                                      [6.09364223e+00, -2.01337896e-02, 4.09343690e-02, -1.33392334e-01],
                                      [6.34268856e+00, 4.24007326e-02, 2.30092421e-01, -1.48071289e-01],
                                      [5.94284391e+00, -3.76656875e-02, -3.89980823e-02, -1.01837158e-01]],
                                     dtype="float32")

    def assert_kernels_not_changed(self, tol=1e-5):
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
        assert_array_almost_equal(center_new, center_old, tol=1e-5)

        # check the method used for the projections
        k = KernelPCA(4)
        k.fit(self.K_tr)
        # copy needed, because the class is doing that elsewhere
        # noinspection PyProtectedMember
        center_method_for_tests = k._center_test_kmat(self.K_tr.copy())
        assert_array_almost_equal(center_method_for_tests, center_old, tol=1e-5)

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
        assert_array_almost_equal(pvec_train, self.ref_pvec_train, tol=5e-4)

        # transform the test
        pvec_test = kpca_obj.transform(self.K_test)
        self.assert_kernels_not_changed()
        assert_array_almost_equal(pvec_test, self.ref_pvec_test, tol=5e-4)

    def test_fit_transform(self):
        # init, fit & transform
        kpca_obj = KernelPCA(4)
        pvec_train = kpca_obj.fit_transform(self.K_tr)
        self.assert_kernels_not_changed()

        # assert that it raises some errors
        self.assert_local_errors(kpca_obj)

        # check result
        assert_array_almost_equal(pvec_train, self.ref_pvec_train, 5e-4)

        # transform the test
        pvec_test = kpca_obj.transform(self.K_test)
        self.assert_kernels_not_changed()
        assert_array_almost_equal(pvec_test, self.ref_pvec_test, tol=5e-4)

    def assert_local_errors(self, kpca_obj):
        # fixme: adthe shape and tye errors as well
        with raises(RuntimeError):
            _pvec_dummy = kpca_obj.fit(self.K_tr)
            self.assert_kernels_not_changed()
        with raises(RuntimeError):
            _pvec_dummy = kpca_obj.fit_transform(self.K_tr)
