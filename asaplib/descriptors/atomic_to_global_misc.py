class Atomic_2_Global_Kernel:
    """
    a simple class to record the kernel functions used to convert atomic descriptors into global ones
    """
    def __init__(self, kernel_type='average', zeta_list=[1], element_wise=False):
        self.kernel_js = {'kernel_type': kernel_type,  
                          'zeta_list': zeta_list,
                          'elementwise': element_wise}
        print("Use kernel functions to compute global descriptors: ", self.kernel_js)
        self.kernel_tag = "-z-"+str(zeta_list)+"-k-"+str(kernel_type)
        if element_wise: self.kernel_tag+="-e" 
        if kernel_type == 'average' and element_wise == False and len(zeta_list)==1 and zeta_list[0]==1:
            # this is the vanilla situation. We just take the average soap for all atoms
            self.kernel_tag = ''

    def get_dict(self):
        return self.kernel_js
    def get_tag(self):
        return self.kernel_tag
    def get(self):
        return self.kernel_js, self.kernel_tag

