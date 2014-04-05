package com.github.neuralnetworks.calculation.test;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.List;

import org.junit.Before;
import org.junit.Test;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.ValuesProvider;

public class ValuesProviderTest {

    private ValuesProvider provider;

    @Before
    public void setup() {
        provider = new ValuesProvider();
    }

    @Test
    public void addANewLayer() {
        Matrix matrix = new Matrix(new float[] { (float) 0.1, (float) 0.2, (float) 0.3, (float) 0.4 }, 2);
        Layer layer = new Layer();

        provider.addValues(layer, matrix);
        assertEquals(matrix, provider.getValues(layer, 2));
    }

    @Test
    public void addANewLayerReplacesOldLayer() {
        Matrix matrix = new Matrix(new float[] { (float) 0.1, (float) 0.2, (float) 0.3, (float) 0.4 }, 2);
        Layer layer = new Layer();

        provider.addValues(layer, matrix);
        provider.addValues(layer, matrix);
        assertEquals(matrix, provider.getValues(layer, 2));
    }

    @Test
    public void addANewLayerAddsUniqueLayer() {
        Matrix matrix = new Matrix(new float[] { (float) 0.1, (float) 0.2, (float) 0.3, (float) 0.4 }, 2);
        Matrix matrix2 = new Matrix(new float[] { (float) 0.1, (float) 0.2, (float) 0.3, (float) 0.4 }, 2);
        Layer layer = new Layer();
        Layer layer2 = new Layer();

        provider.addValues(layer, matrix);
        provider.addValues(layer2, matrix2);
        assertEquals(matrix, provider.getValues(layer, 2));
        assertEquals(matrix2, provider.getValues(layer2, 2));
    }

    @Test
    public void addANewLayerAddsUniqueLayerWithDifferentSize() {
        Matrix matrix = new Matrix(new float[] { (float) 0.1, (float) 0.2, (float) 0.3, (float) 0.4 }, 2);
        Matrix matrix2 = new Matrix(new float[] { (float) 0.1, (float) 0.2, (float) 0.3, (float) 0.4 }, 1);
        Layer layer = new Layer();

        provider.addValues(layer, matrix);
        provider.addValues(layer, matrix2);
        assertEquals(matrix2, provider.getValues(layer, 4));
    }

    @Test
    public void addANewLayerWhosRowIsIncorrectCreatesANewMatrix() {
        Matrix matrix = new Matrix(new float[] { (float) 0.1, (float) 0.2, (float) 0.3, (float) 0.4 }, 1);
        Matrix matrix2 = new Matrix(new float[] { (float) 0.1, (float) 0.2, (float) 0.3, (float) 0.4 }, 2);
        Layer layer = new Layer();

        provider.addValues(layer, matrix);
        provider.addValues(layer, matrix2);

        Matrix newMatrix = provider.getValues(layer, 4);
        assertEquals(4, newMatrix.getRows());
        assertEquals(2, newMatrix.getColumns());
    }

    @Test
    public void getColumnsReturnsZeroWithoutAnyValues() {
        assertEquals(0, provider.getColumns());
    }

    @Test
    public void getColumnsReturnsCorrectColumnsWithSingleMatrix() {
        Matrix matrix = new Matrix(new float[] { (float) 0.1, (float) 0.2, (float) 0.3, (float) 0.4 }, 2);
        Layer layer = new Layer();

        provider.addValues(layer, matrix);
        assertEquals(2, provider.getColumns());
    }

    @Test
    public void getValuesWithANewLayerCreatesANewMatrix() {
        Matrix matrix = new Matrix(new float[] { (float) 0.1, (float) 0.2, (float) 0.3, (float) 0.4 }, 2);
        Layer layer = new Layer();

        provider.addValues(layer, matrix);

        Matrix result = provider.getValues(new Layer(), 1);
        assertEquals(1, result.getRows());
        assertEquals(2, result.getColumns());
    }

    @Test(expected = IllegalArgumentException.class)
    public void getUnitCountThrowsErrorWhenLayerIsntInTheConnection() {
        List<Connections> connections = new ArrayList<Connections>();
        Conv2DConnection connection = new Conv2DConnection(new Layer(), new Layer(), 5, 5, 1, 1, 1, 1, 2);
        connections.add(connection);

        provider.getUnitCount(new Layer(), connections);
    }

    @Test
    public void getUnitCountGetsInput() {
        Layer inputLayer = new Layer();
        Layer outputLayer = new Layer();

        List<Connections> connections = new ArrayList<Connections>();
        Conv2DConnection connection = new Conv2DConnection(inputLayer, outputLayer, 5, 5, 1, 1, 1, 1, 2);
        connections.add(connection);

        assertEquals(connection.getInputUnitCount(), provider.getUnitCount(inputLayer, connections));
    }

    @Test
    public void getUnitCountGetsOutput() {
        Layer inputLayer = new Layer();
        Layer outputLayer = new Layer();

        List<Connections> connections = new ArrayList<Connections>();
        Conv2DConnection connection = new Conv2DConnection(inputLayer, outputLayer, 5, 5, 1, 1, 1, 1, 2);
        connections.add(connection);

        assertEquals(connection.getOutputUnitCount(), provider.getUnitCount(outputLayer, connections));
    }

    @Test
    public void getUnitCountGetsInputWithMultipleConnections() {
        Layer inputLayer = new Layer();
        Layer outputLayer = new Layer();

        List<Connections> connections = new ArrayList<Connections>();
        Conv2DConnection connection = new Conv2DConnection(inputLayer, outputLayer, 5, 5, 1, 1, 1, 1, 2);
        Conv2DConnection connection2 = new Conv2DConnection(inputLayer, outputLayer, 5, 5, 1, 1, 1, 1, 2);
        connections.add(connection);
        connections.add(connection2);

        assertEquals(connection.getInputUnitCount(), provider.getUnitCount(inputLayer, connections));
    }

    @Test
    public void getUnitCountGetsOutputWithMultipleConnections() {
        Layer inputLayer = new Layer();
        Layer outputLayer = new Layer();

        List<Connections> connections = new ArrayList<Connections>();
        Conv2DConnection connection = new Conv2DConnection(inputLayer, outputLayer, 5, 5, 1, 1, 1, 1, 2);
        Conv2DConnection connection2 = new Conv2DConnection(inputLayer, outputLayer, 5, 5, 1, 1, 1, 1, 2);
        connections.add(connection);
        connections.add(connection2);

        assertEquals(connection.getOutputUnitCount(), provider.getUnitCount(outputLayer, connections));
    }

    @Test(expected = IllegalArgumentException.class)
    public void getUnitCountThrowsErrorWithMultipleConnectionsWithDifferentInputUnitCounts() {
        Layer inputLayer = new Layer();
        Layer outputLayer = new Layer();

        List<Connections> connections = new ArrayList<Connections>();
        Conv2DConnection connection = new Conv2DConnection(inputLayer, outputLayer, 5, 5, 1, 1, 1, 1, 2);
        Conv2DConnection connection2 = new Conv2DConnection(inputLayer, outputLayer, 5, 5, 5, 5, 5, 5, 5);
        connections.add(connection);
        connections.add(connection2);

        assertEquals(connection.getInputUnitCount(), provider.getUnitCount(inputLayer, connections));
    }

    @Test(expected = IllegalArgumentException.class)
    public void getUnitCountThrowsErrorWithMultipleConnectionsWithDifferentOutputUnitCounts() {
        Layer inputLayer = new Layer();
        Layer outputLayer = new Layer();

        List<Connections> connections = new ArrayList<Connections>();
        Conv2DConnection connection = new Conv2DConnection(inputLayer, outputLayer, 5, 5, 1, 1, 1, 1, 2);
        Conv2DConnection connection2 = new Conv2DConnection(inputLayer, outputLayer, 5, 5, 5, 5, 5, 5, 5);
        connections.add(connection);
        connections.add(connection2);

        assertEquals(connection.getOutputUnitCount(), provider.getUnitCount(outputLayer, connections));
    }

    @Test
    public void getUnitCountGetsInputWithoutUsingAList() {
        Layer inputLayer = new Layer();
        Layer outputLayer = new Layer();
        Conv2DConnection connection = new Conv2DConnection(inputLayer, outputLayer, 5, 5, 1, 1, 1, 1, 2);

        assertEquals(connection.getInputUnitCount(), provider.getUnitCount(inputLayer, connection));
    }

    @Test
    public void getValuesWithASingleConnection() {
        Layer layer = new Layer();
        Conv2DConnection connection = new Conv2DConnection(layer, new Layer(), 4, 1, 1, 1, 1, 1, 2);

        Matrix matrix = new Matrix(new float[] { (float) 0.1, (float) 0.2, (float) 0.3, (float) 0.4 }, 1);

        provider.addValues(layer, matrix);

        assertEquals(matrix, provider.getValues(layer, connection));
    }

    @Test
    public void getValuesWithJustALayer() {
        Layer layer = new Layer();
        Conv2DConnection connection = new Conv2DConnection(layer, new Layer(), 4, 1, 1, 1, 1, 1, 2);
        layer.addConnection(connection);

        Matrix matrix = new Matrix(new float[] { (float) 0.1, (float) 0.2, (float) 0.3, (float) 0.4 }, 1);
        provider.addValues(layer, matrix);

        assertEquals(matrix, provider.getValues(layer));
    }
}
