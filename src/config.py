gen_dilations = [[1, 1], [3, 1], [5, 1]]
gen_kernel_sizes = [16, 16, 4, 4]

subpd_kernel_size = (5, 1)
subpd_stride = (3, 1)
subpd_channels = [1, 32, 128, 512, 1024]
subpd_padding = (2, 0)

# https://arxiv.org/pdf/1910.06711.pdf + https://github.com/jik876/hifi-gan
subsd_kernels = [15] + [41] * 5 + [5]
subsd_strides = [1, 2, 2, 4, 4, 1, 1]
subsd_groups = [1, 4] + [16] * 4 + [1]
subsd_channels = [1, 128, 128, 256, 512, 1024, 1024]
