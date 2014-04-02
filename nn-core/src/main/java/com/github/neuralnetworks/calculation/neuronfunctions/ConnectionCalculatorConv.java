package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.List;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.ValuesProvider;
import com.github.neuralnetworks.util.Util;

/**
 * Default implementation of Connection calculator for convolutional/subsampling
 * layers
 */
public class ConnectionCalculatorConv implements ConnectionCalculator {

    private static final long serialVersionUID = -5405654469496055017L;

    protected AparapiConv2D inputFunction;
    protected Layer currentLayer;
    protected int miniBatchSize;

    @Override
    public void calculate(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer) {
        Conv2DConnection c = null;
        Conv2DConnection bias = null;

        for (Connections con : connections) {
            if (con instanceof Conv2DConnection) {
                if (Util.isBias(con.getInputLayer())) {
                    bias = (Conv2DConnection) con;
                } else {
                    c = (Conv2DConnection) con;
                }
            }
        }

        if (c != null) {
            // currently works only as a feedforward (including bp)
            if (inputFunction == null || miniBatchSize != valuesProvider.getColumns()) {
                miniBatchSize = valuesProvider.getColumns();
                inputFunction = createInputFunction(c, miniBatchSize);
            }

            calculateBias(bias, valuesProvider);

            if (targetLayer == c.getOutputLayer()) {
                inputFunction.calculate(c, valuesProvider.getValues(Util.getOppositeLayer(c, targetLayer), c), valuesProvider.getValues(targetLayer, c));
            } else {
                inputFunction.calculate(c, valuesProvider.getValues(targetLayer, c), valuesProvider.getValues(Util.getOppositeLayer(c, targetLayer), c));
            }
        }
    }

    protected AparapiConv2D createInputFunction(Conv2DConnection c, int miniBatchSize) {
        return new AparapiConv2DFF(c, miniBatchSize);
    }

    protected void calculateBias(Conv2DConnection bias, ValuesProvider vp) {
        if (bias != null) {
            float[] biasValue = vp.getValues(bias.getInputLayer(), bias).getElements();
            if (biasValue[0] == 0) {
                Util.fillArray(biasValue, 1);
            }

            float[] a = vp.getValues(bias.getOutputLayer(), bias).getElements();
            int fm = a.length / bias.getWeights().length;
            for (int i = 0; i < a.length; i++) {
                a[i] += bias.getWeights()[i / fm];
            }
        }
    }

    public AparapiConv2D getInputFunction() {
        return inputFunction;
    }

    public void setInputFunction(AparapiConv2D inputFunction) {
        this.inputFunction = inputFunction;
    }
}
