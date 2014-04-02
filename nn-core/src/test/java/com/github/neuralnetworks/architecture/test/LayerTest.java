package com.github.neuralnetworks.architecture.test;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.List;

import org.junit.Before;
import org.junit.Test;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;

public class LayerTest {

    private Layer layer;

    @Before
    public void setup() {
        layer = new Layer();
    }

    @Test
    public void newLayerHasEmptyConnectionsList() {
        assertEquals(0, layer.getConnections().size());
    }

    @Test
    public void addAConnection() {
        layer.addConnection(new FullyConnected(new Layer(), new Layer(), 0, 0));

        assertEquals(1, layer.getConnections().size());
    }

    @Test
    public void addAListOfConnection() {
        List<Connections> connectionList = new ArrayList<Connections>();

        connectionList.add(new FullyConnected(new Layer(), new Layer(), 0, 0));
        connectionList.add(new FullyConnected(new Layer(), new Layer(), 0, 0));
        connectionList.add(new FullyConnected(new Layer(), new Layer(), 0, 0));
        connectionList.add(new FullyConnected(new Layer(), new Layer(), 0, 0));

        layer.setConnections(connectionList);

        assertEquals(4, layer.getConnections().size());
    }

    @Test
    public void emptyNeuralNetworkReturnsEmptyList() {
        assertEquals(0, layer.getConnections(new NeuralNetworkImpl()).size());
    }

    @Test
    public void neuralNetworkWithoutCommonConnectionReturnsEmptyList() {
        NeuralNetworkImpl network = new NeuralNetworkImpl();
        network.addLayer(new Layer());

        layer.addConnection(new FullyConnected(new Layer(), new Layer(), 0, 0));

        assertEquals(0, layer.getConnections(network).size());
    }

    @Test
    public void neuralNetworkWithSharedInputConnectedLayer() {
        Layer connectedLayer = new Layer();

        NeuralNetworkImpl network = new NeuralNetworkImpl();
        network.addLayer(connectedLayer);

        layer.addConnection(new FullyConnected(connectedLayer, new Layer(), 0, 0));

        assertEquals(1, layer.getConnections(network).size());
    }

}
