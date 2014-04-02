package com.github.neuralnetworks.calculation.neuronfunctions;

import com.github.neuralnetworks.architecture.Conv2DConnection;

/**
 * Tanh convolutional calculator
 */
public class AparapiConv2DTanh extends ConnectionCalculatorConv {

    private static final long serialVersionUID = -5265802399836825652L;

    @Override
    protected AparapiConv2D createInputFunction(Conv2DConnection c, int miniBatchSize) {
        return new AparapiConv2DTanhFunction(c, miniBatchSize);
    }

    public static class AparapiConv2DTanhFunction extends AparapiConv2DFF {

        private static final long serialVersionUID = -7985734201416578973L;

        public AparapiConv2DTanhFunction(Conv2DConnection c, int miniBatchSize) {
            super(c, miniBatchSize);
        }

        @Override
        protected float activationFunction(float value) {
            return tan(value);
        }
    }
}
