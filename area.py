#
# Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt
#

import numpy as np


def make_index_along_axis(idx, axis, ndims):
    axis = axis % ndims
    return tuple([idx if dim == axis else slice(None) for dim in range(ndims)])


def rescale_area(x, siz_out, axis=0):
    siz_in = x.shape[axis]
    ndims = len(x.shape)
    siz_out = int(siz_out)

    # start
    price = 2*siz_out
    budget = 2*siz_in
    carry = siz_in + siz_out
    index_in = 0
    while carry > price:
        index_in = index_in - 1
        carry = carry - price

    # ciclying
    y_shape = list(x.shape)
    y_shape[axis] = siz_out
    e_shape = list(x.shape)
    del e_shape[axis]
    y = np.empty(y_shape, dtype=x.dtype)

    last_x = x[make_index_along_axis(
        index_in % siz_in, axis=axis, ndims=ndims)]

    for index_out in range(siz_out):
        wallet = budget
        e = np.zeros(e_shape, dtype=x.dtype)
        while wallet > 0:
            if wallet >= carry:
                e = e + carry * last_x
                wallet = wallet - carry
                index_in = index_in + 1
                last_x = x[make_index_along_axis(
                    index_in % siz_in, axis=axis, ndims=ndims)]
                carry = price
            else:
                e = e + wallet * last_x
                carry = carry - wallet
                wallet = 0
                break
        y[make_index_along_axis(index_out, axis=axis, ndims=ndims)] = e/budget

    return y
