package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.List;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.ValuesProvider;

/**
 * Max pooling
 */
public class AparapiMaxPooling2D implements ConnectionCalculator {

    private static final long serialVersionUID = 8165829315701496713L;

    private AparapiMaxPooling2DCC cc;

    @Override
    public void calculate(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer) {
        if (cc == null || cc.getMiniBatchSize() != valuesProvider.getColumns()) {
            cc = new AparapiMaxPooling2DCC((Subsampling2DConnection) connections.get(0), valuesProvider.getColumns());
        }

        cc.calculate(connections, valuesProvider, targetLayer);
    }

    public static class AparapiMaxPooling2DCC extends AparapiSubsampling2D {

        private static final long serialVersionUID = -2393526660090301257L;

        public AparapiMaxPooling2DCC(Subsampling2DConnection c, int miniBatchSize) {
            super(c, miniBatchSize);
        }

        @Override
        protected void pool(int inputStartIndex) {
            int rl = regionLength;
            int miniBatch = miniBatchSize;
            float max = 0;

            for (int i = 0; i < miniBatch; i++) {
                max = input[(inputStartIndex + featureMapOffsets[0]) * miniBatch + i];
                for (int j = 1; j < rl; j++) {
                    max = max(input[(inputStartIndex + featureMapOffsets[j]) * miniBatch + i], max);
                }

                output[getGlobalId() * miniBatch + i] = max;
            }
        }
    }
}
