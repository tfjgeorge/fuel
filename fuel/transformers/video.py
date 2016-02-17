import numpy
from PIL import Image
from . import ExpectsAxisLabels, Transformer
import math


class RescaleMinDimension(Transformer, ExpectsAxisLabels):
    """Resize (lists of) images so that their shortest dimension is of a given size.

    Parameters
    ----------
    data_stream : instance of :class:`AbstractDataStream`
        The data stream to wrap.
    min_dimension_size : int
        The desired length of the smallest dimension.
    resample : str, optional
        Resampling filter for PIL to use to upsample any images requiring
        it. Options include 'nearest' (default), 'bilinear', and 'bicubic'.
        See the PIL documentation for more detailed information.

    Notes
    -----
    This transformer expects stream sources returning individual images,
    represented as 2- or 3-dimensional arrays, or lists of the same.
    The format of the stream is unaltered.

    """
    def __init__(self, data_stream, min_dimension_size, resample='nearest',
                 **kwargs):
        self.min_dimension_size = min_dimension_size
        try:
            self.resample = getattr(Image, resample.upper())
        except AttributeError:
            raise ValueError("unknown resampling filter '{}'".format(resample))
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(RescaleMinDimension, self).__init__(data_stream, **kwargs)

    def transform_batch(self, batch):
        # self.verify_axis_labels(('batch', 'channel', 'height', 'width'),
        #                         self.data_stream.axis_labels[source_name],
        #                         source_name)<
        output = ([],[],[])
        for case, images, targets in zip(batch[0], batch[1], batch[2]):
            output[0].append(case)
            rescaled_imgs, rescaled_tgt = self._example_transform(images, targets) 
            output[1].append(rescaled_imgs)
            output[2].append(rescaled_tgt)

        return output

    def transform_example(self, example):
        # self.verify_axis_labels(('channel', 'height', 'width'),
        #                         self.data_stream.axis_labels[source_name],
        #                         source_name)
        return self._example_transform(example)

    def _example_transform(self, example, target_sizes):
        if example.ndim > 3 or example.ndim < 2:
            raise NotImplementedError
        original_height, original_width = example.shape[-2:]
        multiplier = max(float(self.min_dimension_size) / original_width, float(self.min_dimension_size) / original_height)

        width = int(math.ceil(original_width * multiplier))
        height = int(math.ceil(original_height * multiplier))

        dt = example.dtype
        target = numpy.zeros((example.shape[0], height, width))

        for i in range(example.shape[0]):

            im = Image.fromarray(example[i,:,:].astype('int16'))
            im = numpy.array(im.resize((width, height))).astype(dt)

            target[i,:,:] = im
        return target, target_sizes*multiplier
