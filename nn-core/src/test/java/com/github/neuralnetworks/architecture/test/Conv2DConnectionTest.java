package com.github.neuralnetworks.architecture.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import org.junit.Before;
import org.junit.Test;

import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.Layer;

public class Conv2DConnectionTest {

    private final Layer inputLayer = new Layer();
    private final Layer outputLayer = new Layer();

    private Conv2DConnection connection;
    int inputFeatureMapColumns = 10;
    int inputFeatureMapRows = 10;
    int inputFilters = 2;
    int kernelRows = 2;
    int kernelColumns = 2;
    int outputFilters = 2;
    int stride = 5;

    @Before
    public void setup() {
        connection = new Conv2DConnection(inputLayer, outputLayer, inputFeatureMapColumns, inputFeatureMapRows,
                inputFilters, kernelRows, kernelColumns, outputFilters, stride);
    }

    @Test
    public void testUpdateDimensions() {
        connection.updateDimensions();

        float[] weights = connection.getWeights();
        assertEquals(kernelColumns * kernelRows * outputFilters * inputFilters, weights.length);
        assertEquals(0, weights[0], 0);
    }

    @Test
    public void testUpdateDimensionsCreatesNewWeightsWithIncorrectArraySize() {
        float[] weights = new float[] { (float) 0.0, (float) 1.1, (float) 2.2, (float) 3.3 };
        connection.setWeights(weights);
        connection.updateDimensions();

        assertEquals(kernelColumns * kernelRows * outputFilters * inputFilters, connection.getWeights().length);
        assertEquals(0, weights[0], 0);
    }

    @Test
    public void testUpdateDimensionsDoesntCreateNewWeightsWithCorrectArraySize() {
        float[] weights = new float[kernelColumns * kernelRows * outputFilters * inputFilters];
        for (int i = 0; i < kernelColumns * kernelRows * outputFilters * inputFilters; i++) {
            weights[i] = (float) 0.5;
        }

        connection.setWeights(weights);
        connection.updateDimensions();

        assertEquals(kernelColumns * kernelRows * outputFilters * inputFilters, connection.getWeights().length);
        assertEquals(0.5, weights[0], 0);
    }

    @Test
    public void testSetWeights() {
        float[] weights = new float[] { (float) 0.0, (float) 1.1, (float) 2.2, (float) 3.3 };
        connection.setWeights(weights);

        assertEquals(weights, connection.getWeights());
    }

    @Test
    public void testGetKernelRows() {
        assertEquals(kernelRows, connection.getKernelRows());
    }

    @Test
    public void testGetKernelColumns() {
        assertEquals(kernelColumns, connection.getKernelColumns());
    }

    @Test
    public void testGetInputUnitCount() {
        assertEquals(inputFeatureMapRows * inputFeatureMapColumns * inputFilters, connection.getInputUnitCount());
    }

    @Test
    public void testGetOutputUnitCount() {
        assertEquals(connection.getOutputFeatureMapLength() * outputFilters, connection.getOutputUnitCount());
    }

    @Test
    public void testGetInputFeatureMapColumns() {
        assertEquals(inputFeatureMapColumns, connection.getInputFeatureMapColumns());
    }

    @Test
    public void testSetInputFeatureMapColumns() {
        int newInputFeatureMapColumns = 1000;
        connection.setInputFeatureMapColumns(newInputFeatureMapColumns);

        assertEquals(newInputFeatureMapColumns, connection.getInputFeatureMapColumns());
    }

    @Test
    public void testGetInputFeatureMapRows() {
        assertEquals(inputFeatureMapRows, connection.getInputFeatureMapRows());
    }

    @Test
    public void testSetInputFeatureMapRows() {
        int newInputFeatureMapRows = 1000;
        connection.setInputFeatureMapRows(newInputFeatureMapRows);

        assertEquals(newInputFeatureMapRows, connection.getInputFeatureMapRows());
    }

    @Test
    public void testGetInputFeatureMapLength() {
        assertEquals(inputFeatureMapRows * inputFeatureMapColumns, connection.getInputFeatureMapLength());
    }

    @Test
    public void testGetOutputFeatureMapLength() {
        assertEquals(connection.getOutputFeatureMapRows() * connection.getOutputFeatureMapColumns(),
                connection.getOutputFeatureMapLength());
    }

    @Test
    public void testGetInputFilters() {
        assertEquals(inputFilters, connection.getInputFilters());
    }

    @Test
    public void testSetInputFilters() {
        int newInputFilters = 1000;
        connection.setInputFilters(newInputFilters);

        assertEquals(newInputFilters, connection.getInputFilters());
    }

    @Test
    public void testGetOutputFeatureMapColumns() {
        assertEquals((inputFeatureMapRows - kernelRows) / stride + 1, connection.getOutputFeatureMapColumns());
    }

    @Test
    public void testGetOutputFeatureMapRows() {
        assertEquals((inputFeatureMapColumns - kernelColumns) / stride + 1, connection.getOutputFeatureMapRows());
    }

    @Test
    public void testGetOutputFilters() {
        assertEquals(outputFilters, connection.getOutputFilters());
    }

    @Test
    public void testSetOutputFilters() {
        int newOutputFilters = 1000;
        connection.setOutputFilters(newOutputFilters);

        assertEquals(newOutputFilters, connection.getOutputFilters());
    }

    @Test
    public void testGetStride() {
        assertEquals(stride, connection.getStride());
    }

    @Test
    public void testSetStride() {
        int newStride = 1000;
        connection.setStride(newStride);

        assertEquals(newStride, connection.getStride());
    }

    @Test
    public void testGetInputLayer() {
        assertEquals(inputLayer, connection.getInputLayer());
    }

    @Test
    public void testSetInputLayer() {
        Layer newInputLayer = new Layer();
        connection.setInputLayer(newInputLayer);

        assertEquals(newInputLayer, connection.getInputLayer());
    }

    @Test
    public void testGetOutputLayer() {
        assertEquals(outputLayer, connection.getOutputLayer());
    }

    @Test
    public void testSetOutputLayer() {
        Layer newOutputLayer = new Layer();
        connection.setOutputLayer(newOutputLayer);

        assertEquals(newOutputLayer, connection.getOutputLayer());
    }

    @Test
    public void testGetLayers() {
        assertEquals(2, connection.getLayers().size());
    }

    @Test
    public void testGetConnections() {
        assertTrue(connection.getConnections().contains(connection));
    }

    @Test
    public void testGetLayerCalculator() {
        assertNull(connection.getLayerCalculator());
    }

    @Test
    public void testCompareToWithEqualConnections() {
        assertEquals(0, connection.compareTo(connection));
    }

}
