package com.github.neuralnetworks.util.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.util.Util;

public class UtilTest {

    private final Layer inputLayer = new Layer();
    private final Layer outputLayer = new Layer();
    private final List<Connections> connections = new ArrayList<Connections>();

    @Test
    public void fillFloatArray() {
        float[] array = new float[10];
        Util.fillArray(array, (float) 0.1);

        for (float f : array) {
            assertEquals(0.1, f, 0.1);
        }
    }

    @Test
    public void doesntFillEmptyFloatArray() {
        float[] array = new float[0];
        Util.fillArray(array, (float) 0.1);

        assertEquals(0, array.length);
    }

    @Test
    public void fillIntArray() {
        int[] array = new int[10];
        Util.fillArray(array, 1);

        for (int f : array) {
            assertEquals(1, f);
        }
    }

    @Test
    public void doesntFillEmptyIntArray() {
        int[] array = new int[0];
        Util.fillArray(array, 1);

        assertEquals(0, array.length);
    }

    private Conv2DConnection getConv2DConnection() {
        int inputFeatureMapColumns = 20;
        int inputFeatureMapRows = 10;
        int inputFilters = 1;
        int kernelRows = 1;
        int kernelColumns = 1;
        int outputFilters = 2;
        int stride = 1;

        return new Conv2DConnection(inputLayer, outputLayer, inputFeatureMapColumns, inputFeatureMapRows, inputFilters,
                kernelRows, kernelColumns, outputFilters, stride);
    }

    private Subsampling2DConnection getSubsampling2DConnection() {
        int inputFeatureMapColumns = 20;
        int inputFeatureMapRows = 10;
        int subsamplingRegionRows = 1;
        int subsamplingRegionCols = 1;
        int filters = 1;

        return new Subsampling2DConnection(inputLayer, outputLayer, inputFeatureMapColumns, inputFeatureMapRows,
                subsamplingRegionRows, subsamplingRegionCols, filters);
    }

    private FullyConnected getFullyConnected() {
        return new FullyConnected(inputLayer, outputLayer, 0, 0);
    }

    @Test
    public void getOppositeLayer() {
        Conv2DConnection connection = getConv2DConnection();

        assertEquals(inputLayer, Util.getOppositeLayer(connection, outputLayer));
        assertEquals(outputLayer, Util.getOppositeLayer(connection, inputLayer));
    }

    @Test
    public void layerWithMultipleConnectionsIsNotABiasLayer() {
        Layer topLayer = new Layer();
        Layer middleLayer = new Layer();
        Layer bottomLayer = new Layer();

        List<Connections> connections = new ArrayList<Connections>();
        Connections firstConnection = new FullyConnected(topLayer, middleLayer, 0, 0);
        connections.add(firstConnection);

        Connections secondConnection = new FullyConnected(topLayer, bottomLayer, 0, 0);
        connections.add(secondConnection);

        topLayer.setConnections(connections);

        assertFalse(Util.isBias(topLayer));
    }

    @Test
    public void conv2DConnectionIsBiasLayer() {
        Layer inputLayer = new Layer();
        Layer outputLayer = new Layer();
        int inputFeatureMapColumns = 10;
        int inputFeatureMapRows = 10;
        int inputFilters = 1;
        int kernelRows = 1;
        int kernelColumns = 1;
        int outputFilters = 2;
        int stride = 1;

        Conv2DConnection connection = new Conv2DConnection(inputLayer, outputLayer, inputFeatureMapColumns,
                inputFeatureMapRows, inputFilters, kernelRows, kernelColumns, outputFilters, stride);
        connections.add(connection);
        inputLayer.setConnections(connections);

        assertTrue(Util.isBias(inputLayer));
    }

    @Test
    public void conv2DConnectionWithTooManyInputFiltersIsntBiasLayer() {
        connections.add(getConv2DConnection());
        inputLayer.setConnections(connections);

        assertFalse(Util.isBias(inputLayer));
    }

    @Test
    public void conv2DConnectionWithIncorrectInputColumsIsntBiasLayer() {
        connections.add(getConv2DConnection());
        inputLayer.setConnections(connections);

        assertFalse(Util.isBias(inputLayer));
    }

    @Test
    public void fullyConnectedGraphIsBiasLayer() {
        Connections connection = new FullyConnected(inputLayer, outputLayer, 1, 0);
        connections.add(connection);
        inputLayer.setConnections(connections);

        assertTrue(Util.isBias(inputLayer));
    }

    @Test
    public void fullyConnectedGraphWithTooManyColumnsIsntBiasLayer() {
        connections.add(getFullyConnected());
        inputLayer.setConnections(connections);

        assertFalse(Util.isBias(inputLayer));
    }

    @Test
    public void outputLayerIsntBiasLayer() {
        connections.add(getFullyConnected());
        inputLayer.setConnections(connections);

        assertFalse(Util.isBias(outputLayer));
    }

    @Test
    public void subsampleConnectionIsntBiasLayer() {
        connections.add(getSubsampling2DConnection());
        inputLayer.setConnections(connections);

        assertFalse(Util.isBias(inputLayer));
    }

    @Test
    public void layerWithAConv2DConnectionIsntASubsamplingLayer() {
        connections.add(getConv2DConnection());
        inputLayer.setConnections(connections);

        assertFalse(Util.isSubsampling(inputLayer));
    }

    @Test
    public void subsampling2DConnectionIsASubsamplingLayer() {
        connections.add(getSubsampling2DConnection());
        inputLayer.setConnections(connections);

        assertTrue(Util.isSubsampling(outputLayer));
    }

    @Test
    public void subsampling2DConnectionWithConv2DConnectionIsASubsamplingLayer() {
        connections.add(getSubsampling2DConnection());
        inputLayer.setConnections(connections);

        assertTrue(Util.isSubsampling(inputLayer));
    }

    @Test
    public void subsampling2DConnectionWithInputLayerIsntASubsamplingLayer() {
        connections.add(getSubsampling2DConnection());
        connections.add(getConv2DConnection());
        inputLayer.setConnections(connections);

        assertFalse(Util.isSubsampling(inputLayer));
    }

    @Test
    public void fullyConnectedGraphIsntASubsamplingLayer() {
        connections.add(getFullyConnected());
        inputLayer.setConnections(connections);

        assertFalse(Util.isSubsampling(inputLayer));
    }

    @Test
    public void subsampleConnectionIsntAConvolutionalLayer() {
        connections.add(getSubsampling2DConnection());
        inputLayer.setConnections(connections);

        assertFalse(Util.isConvolutional(inputLayer));
    }

    @Test
    public void conv2DConnectionWithoutSubSampleIsAConvolutionalLayer() {
        connections.add(getConv2DConnection());
        inputLayer.setConnections(connections);

        assertTrue(Util.isConvolutional(outputLayer));
    }

    @Test
    public void conv2DConnectionWithSubSampleIsAConvolutionalLayer() {
        connections.add(getConv2DConnection());
        connections.add(getSubsampling2DConnection());
        inputLayer.setConnections(connections);

        assertTrue(Util.isConvolutional(outputLayer));
    }

    @Test
    public void conv2DConnectionWithInputLayerIsntAConvolutionalLayer() {
        connections.add(getConv2DConnection());
        connections.add(getSubsampling2DConnection());
        inputLayer.setConnections(connections);

        assertFalse(Util.isConvolutional(inputLayer));
    }

    @Test
    public void fullyConnectedGraphIsntAConvolutionalLayer() {
        connections.add(getFullyConnected());
        inputLayer.setConnections(connections);

        assertFalse(Util.isConvolutional(inputLayer));
    }

    @Test
    public void emptyConnectionsDoesntHaveBias() {
        assertFalse(Util.hasBias(connections));
    }

    @Test
    public void aConnectionHasBias() {
        Connections connection = new FullyConnected(inputLayer, outputLayer, 1, 0);
        connections.add(connection);
        inputLayer.setConnections(connections);

        assertTrue(Util.hasBias(connections));
    }

    @Test
    public void aConnectionDoesntHaveBias() {
        Connections connection = new FullyConnected(inputLayer, outputLayer, 2, 0);
        connections.add(connection);
        inputLayer.setConnections(connections);

        assertFalse(Util.hasBias(connections));
    }
}
