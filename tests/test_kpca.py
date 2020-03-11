import ase.io
import numpy as np
from dscribe.descriptors import SOAP

from asaplib.pca.ml_kpca import KernelPCA

try:
    from . import asap_testobject
except ImportError:
    import asap_testobject


class TestKPCA(asap_testobject.ASAPlibTestCase):
    def setUp(self):
        # todo: add docs for these
        fn = "ice_test.xyz"
        at_train = ase.io.read(fn, '0')
        at_test = ase.io.read(fn, '1')

        soap_desc = SOAP(species=['H', 'O'], rcut=4., nmax=6, lmax=6, sigma=0.3, periodic=True)
        desc_train = soap_desc.create(at_train)
        desc_test = soap_desc.create(at_test)

        self.K_tr = np.dot(desc_train, desc_train.T) ** 2
        self.K_test = np.dot(desc_test, desc_train.T) ** 2

        self.K_tr_save = self.K_tr.copy()
        self.K_test_save = self.K_test.copy()

        # references:
        self.ref_pvec_train = np.array([[-3.2303891e+00, 9.6056849e-01, -3.0840870e-03, -1.2017698e-02],
                                        [-3.2225225e+00, 9.6638560e-01, -1.6610974e-03, 3.2059520e-02],
                                        [-2.8951304e+00, -9.9193245e-01, -7.8823622e-03, -6.9802892e-03],
                                        [-2.8980618e+00, -1.0126923e+00, -9.6745901e-03, -2.6164867e-02],
                                        [-2.8911943e+00, -1.0052572e+00, -8.4379651e-03, -2.1795810e-03],
                                        [-2.8940110e+00, -1.0260904e+00, -1.0213512e-02, -2.1328345e-02],
                                        [-3.2414720e+00, 9.7098809e-01, -3.1033612e-03, -2.3895590e-02],
                                        [-3.2251790e+00, 9.5365107e-01, -1.8045461e-03, 2.5246326e-02],
                                        [-2.8825021e+00, -1.0000941e+00, -8.1770960e-03, 9.1385255e-03],
                                        [-2.8809681e+00, -9.9579626e-01, -8.2781361e-03, 1.9337641e-02],
                                        [-3.2404327e+00, 9.9212706e-01, -1.1223719e-03, -2.2620136e-02],
                                        [-3.2338450e+00, 9.6978092e-01, -1.5512105e-03, -2.3886204e-02],
                                        [-3.2254500e+00, 9.6266723e-01, -1.2437602e-03, 1.1454688e-02],
                                        [-3.2258351e+00, 9.6286207e-01, -1.5204158e-03, 1.3519926e-02],
                                        [-2.8816104e+00, -1.0021194e+00, -8.1492364e-03, 8.5924473e-03],
                                        [-2.8800888e+00, -9.9781585e-01, -8.2540475e-03, 1.9014006e-02],
                                        [6.2599421e+00, 4.6085209e-02, -5.9356409e-01, 1.4184044e-02],
                                        [6.0120721e+00, 3.1208023e-02, 5.9663945e-01, -7.0962436e-02],
                                        [5.9338422e+00, 1.9080747e-02, 6.1807740e-01, -2.4141915e-02],
                                        [6.2599916e+00, 4.7300328e-02, -5.9232450e-01, 1.0204918e-02],
                                        [5.9501839e+00, 2.4574103e-02, 6.2793219e-01, 4.6252709e-02],
                                        [6.3134155e+00, 5.0569963e-02, -6.1183619e-01, -7.6005317e-02],
                                        [6.2682405e+00, 4.9978156e-02, -5.8744663e-01, 5.5363521e-02],
                                        [5.9510040e+00, 2.3956347e-02, 6.2667006e-01, 4.5620188e-02]],
                                       dtype="float32")

        self.ref_pvec_test = np.array([[-3.36862016e+00, 1.39037058e-01, -1.54231757e-01, 1.71058979e-02],
                                       [-3.32468557e+00, -1.34104341e-02, -1.40938178e-01, 1.10897072e-01],
                                       [-3.41949081e+00, 1.65344656e-01, -1.64024517e-01, 7.11742491e-02],
                                       [-3.33795691e+00, 1.53481707e-01, -1.31440684e-01, 1.11587523e-02],
                                       [-3.35102773e+00, 1.59436524e-01, -1.45192146e-01, 2.47181822e-02],
                                       [-3.33585882e+00, 6.37856498e-02, -1.29543394e-01, 3.40880640e-02],
                                       [-3.36398435e+00, 1.08018173e-02, -1.58899188e-01, 9.01430249e-02],
                                       [-3.34487295e+00, 1.76007405e-01, -1.33781523e-01, 5.15416265e-02],
                                       [-3.10268068e+00, 9.11338851e-02, -7.42703974e-02, 1.20135613e-01],
                                       [-3.12647200e+00, 1.06033854e-01, -8.68724659e-02, 1.14861310e-01],
                                       [-2.74729872e+00, -6.39082305e-03, 7.73019046e-02, 1.39645413e-01],
                                       [-2.82603574e+00, -8.71595860e-01, 5.36031574e-02, -1.62527502e-01],
                                       [-3.23301053e+00, 8.73263776e-02, -1.12462252e-01, 7.37995282e-02],
                                       [-2.85862279e+00, -5.97478002e-02, 6.15883321e-02, 1.82083085e-01],
                                       [-3.16856170e+00, 1.96224779e-01, -9.43925902e-02, 1.57774210e-01],
                                       [-2.86860633e+00, -1.02501380e+00, 1.92470253e-02, -1.20704994e-01],
                                       [-3.26253963e+00, 1.27189964e-01, -9.70077291e-02, 8.13410282e-02],
                                       [-2.89024496e+00, 1.41650429e-02, 5.59556149e-02, 1.71658546e-01],
                                       [-3.18666029e+00, 2.40496054e-01, -9.83804539e-02, 1.59390777e-01],
                                       [-2.86944747e+00, -1.04161191e+00, 1.00002252e-02, -6.62439167e-02],
                                       [-3.29412556e+00, 1.81503892e-01, -1.03239127e-01, 7.30859265e-02],
                                       [-2.90105653e+00, 1.54055417e-01, 6.42177984e-02, 1.40368730e-01],
                                       [-3.16238594e+00, 2.64649093e-01, -8.14419985e-02, 1.28457069e-01],
                                       [-2.68499184e+00, -1.00660419e+00, 2.48309895e-02, 4.88048941e-02],
                                       [6.95653868e+00, 1.96824238e-01, -4.27588165e-01, -1.58803225e-01],
                                       [6.99313831e+00, 2.01924101e-01, -4.43690151e-01, -1.97129592e-01],
                                       [6.92610359e+00, 2.01081753e-01, -3.33780229e-01, -1.49187908e-01],
                                       [6.93492746e+00, 1.93236291e-01, -3.75559509e-01, -1.11254431e-01],
                                       [5.96584368e+00, -4.36020084e-02, -3.06649446e-01, -3.94179858e-02],
                                       [6.16355371e+00, 3.33596505e-02, 2.48846710e-01, -5.23653515e-02],
                                       [6.25925446e+00, 2.16478948e-02, -1.12034827e-01, -1.89658895e-01],
                                       [6.11991644e+00, -1.30746411e-02, -5.71816042e-03, -1.28767669e-01],
                                       [6.33158827e+00, 4.06524912e-02, -1.67217538e-01, -1.74810350e-01],
                                       [6.11753702e+00, -1.94333717e-02, -4.30712402e-02, -1.36734307e-01],
                                       [6.36551094e+00, 4.23447825e-02, -2.32789576e-01, -1.51842445e-01],
                                       [5.96289206e+00, -3.70615236e-02, 4.07243147e-02, -1.02457888e-01]],
                                      dtype="float32")

    def assertKernelsNotChanged(self, tol=1e-5):
        self.assertArrayAlmostEqual(self.K_tr, self.K_tr_save, tol=tol)
        self.assertArrayAlmostEqual(self.K_test, self.K_test_save, tol=tol)

    @staticmethod
    def fixcenter(kernel):
        """
        Taken from older versio of ml_kpca.py, As Is, for testing the centering even if that is removed
        """
        cernel = kernel.copy()
        cols = np.mean(cernel, axis=0)
        # print "numcol ", cols.shape
        rows = np.mean(cernel, axis=1)
        # print "numrows", rows.shape
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
        self.assertArrayAlmostEqual(center_new, center_old, tol=1e-5)

    def test_fit_and_transform(self):
        # init & fit
        kpca_obj = KernelPCA(4)
        kpca_obj.fit(self.K_tr)
        self.assertKernelsNotChanged()

        # assert that it raises some errors
        self.assertLocalErrors(kpca_obj)

        # transform
        pvec_train = kpca_obj.transform(self.K_tr)
        self.assertKernelsNotChanged()

        # check result
        self.assertArrayAlmostEqual(pvec_train, self.ref_pvec_train, tol=1e-4)

        # transform the test
        pvec_test = kpca_obj.transform(self.K_test)
        self.assertKernelsNotChanged()
        self.assertArrayAlmostEqual(pvec_test, self.ref_pvec_test)

    def test_fit_transform(self):
        # init, fit & transform
        kpca_obj = KernelPCA(4)
        pvec_train = kpca_obj.fit_transform(self.K_tr)
        self.assertKernelsNotChanged()

        # assert that it raises some errors
        self.assertLocalErrors(kpca_obj)

        # check result
        self.assertArrayAlmostEqual(pvec_train, self.ref_pvec_train)

        # transform the test
        pvec_test = kpca_obj.transform(self.K_test)
        self.assertKernelsNotChanged()
        self.assertArrayAlmostEqual(pvec_test, self.ref_pvec_test)

    def assertLocalErrors(self, kpca_obj):
        # fixme: adthe shape and tye errors as well
        with self.subTest("Errors for fit & fit_transform"):
            with self.assertRaises(RuntimeError):
                pvec_dummy = kpca_obj.fit(self.K_tr)
                self.assertKernelsNotChanged()
            with self.assertRaises(RuntimeError):
                pvec_dummy = kpca_obj.fit_transform(self.K_tr)
