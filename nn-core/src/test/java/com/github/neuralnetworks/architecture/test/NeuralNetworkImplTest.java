package com.github.neuralnetworks.architecture.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.junit.Before;
import org.junit.Test;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;

public class NeuralNetworkImplTest {
    private NeuralNetworkImpl network;

    @Before
    public void setup() {
        network = new NeuralNetworkImpl();
    }

    @Test
    public void uninitializedNetworkHasEmptyLayers() {
        assertEquals(0, network.getLayers().size());
    }

    @Test
    public void uninitializedNetworkDoesntHaveALayerCalculator() {
        assertNull(network.getLayerCalculator());
    }

    @Test
    public void initializedNetworkHasLayers() {
        List<Layer> layers = new ArrayList<Layer>();
        layers.add(new Layer());
        layers.add(new Layer());
        layers.add(new Layer());

        NeuralNetworkImpl network = new NeuralNetworkImpl(layers);
        assertEquals(layers, network.getLayers());
    }

    @Test
    public void canSetLayersOnAnUninitializedNetwork() {
        assertEquals(0, network.getLayers().size());

        Set<Layer> layers = new HashSet<Layer>();
        layers.add(new Layer());
        layers.add(new Layer());
        layers.add(new Layer());

        network.setLayers(layers);
        assertEquals(layers, network.getLayers());
    }

    @Test
    public void returnsInputLayer() {
        Set<Layer> layers = new HashSet<Layer>();
        Layer topLayer = new Layer();
        Layer middleLayer = new Layer();
        Layer bottomLayer = new Layer();

        Connections firstConnection = new FullyConnected(topLayer, middleLayer, 0, 0);
        Connections secondConnection = new FullyConnected(middleLayer, bottomLayer, 0, 0);

        topLayer.addConnection(firstConnection);
        middleLayer.addConnection(firstConnection);

        middleLayer.addConnection(secondConnection);
        bottomLayer.addConnection(secondConnection);

        layers.add(topLayer);
        layers.add(middleLayer);
        layers.add(bottomLayer);
        network.setLayers(layers);

        assertEquals(topLayer, network.getInputLayer());
    }

    @Test
    public void returnsNullInputLayerOnEmptyNetwork() {
        assertNull(network.getInputLayer());
    }

    @Test
    public void returnsOutputLayer() {
        Set<Layer> layers = new HashSet<Layer>();
        Layer topLayer = new Layer();
        Layer middleLayer = new Layer();
        Layer bottomLayer = new Layer();

        Connections firstConnection = new FullyConnected(topLayer, middleLayer, 0, 0);
        Connections secondConnection = new FullyConnected(middleLayer, bottomLayer, 0, 0);

        topLayer.addConnection(firstConnection);
        middleLayer.addConnection(firstConnection);

        middleLayer.addConnection(secondConnection);
        bottomLayer.addConnection(secondConnection);

        layers.add(topLayer);
        layers.add(middleLayer);
        layers.add(bottomLayer);
        network.setLayers(layers);

        assertEquals(bottomLayer, network.getOutputLayer());
    }

    @Test
    public void returnsNullOutputLayerOnEmptyNetwork() {
        assertNull(network.getOutputLayer());
    }

    @Test
    public void returnsConnections() {
        Set<Layer> layers = new HashSet<Layer>();
        Layer topLayer = new Layer();
        Layer middleLayer = new Layer();
        Layer bottomLayer = new Layer();

        Connections firstConnection = new FullyConnected(topLayer, middleLayer, 0, 0);
        Connections secondConnection = new FullyConnected(middleLayer, bottomLayer, 0, 0);

        topLayer.addConnection(firstConnection);
        middleLayer.addConnection(firstConnection);

        middleLayer.addConnection(secondConnection);
        bottomLayer.addConnection(secondConnection);

        layers.add(topLayer);
        layers.add(middleLayer);
        layers.add(bottomLayer);
        network.setLayers(layers);

        List<Connections> connections = network.getConnections();
        assertTrue(connections.contains(firstConnection));
        assertTrue(connections.contains(secondConnection));
    }

    @Test
    public void returnsEmptySetOfConnectionsOnEmptyNetwork() {
        assertEquals(0, network.getConnections().size());
    }
}
