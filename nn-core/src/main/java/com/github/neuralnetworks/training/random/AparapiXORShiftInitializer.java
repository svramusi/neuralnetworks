package com.github.neuralnetworks.training.random;

import java.util.HashMap;
import java.util.Map;

import com.github.neuralnetworks.util.Environment;

public class AparapiXORShiftInitializer implements RandomInitializer {

    private static final long serialVersionUID = 1L;

    protected float start;
    protected float range;
    protected Map<Integer, XORShift> kernels = new HashMap<>();

    public AparapiXORShiftInitializer() {
        super();
        this.start = 0;
        this.range = 1;
    }

    public AparapiXORShiftInitializer(float start, float end) {
        super();
        this.start = start;
        this.range = end - start;
    }

    @Override
    public void initialize(float[] array) {
        XORShift kernel = kernels.get(array.length);
        if (kernel == null) {
            kernels.put(array.length, kernel = new XORShift(array.length, start, range));
        }

        kernel.array = array;

        Environment.getInstance().getExecutionStrategy().execute(kernel, array.length);
    }

    private static class XORShift extends XORShiftKernel {

        private float[] array;
        private final float start;
        private final float range;

        public XORShift(int maximumRange, float start, float range) {
            super(maximumRange);
            this.start = start;
            this.range = range;
        }

        @Override
        public void run() {
            array[getGlobalId()] = start + randomGaussian() * range;
        }
    }
}
