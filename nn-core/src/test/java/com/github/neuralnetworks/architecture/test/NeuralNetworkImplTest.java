package com.github.neuralnetworks.architecture.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
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
import com.github.neuralnetworks.calculation.LayerCalculatorImpl;

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
    public void setLayerCalculator() {
        LayerCalculatorImpl calc = new LayerCalculatorImpl();
        network.setLayerCalculator(calc);

        assertEquals(calc, network.getLayerCalculator());
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

    @Test
    public void returnsEmptySetOfConnectionsOnNullLayers() {
        network.setLayers(null);
        assertEquals(0, network.getConnections().size());
    }

    @Test
    public void returnsConnectionBetweenTwoLayers() {
        Set<Layer> layers = new HashSet<Layer>();
        Layer topLayer = new Layer();
        Layer middleLayer = new Layer();

        Connections firstConnection = new FullyConnected(topLayer, middleLayer, 0, 0);

        topLayer.addConnection(firstConnection);

        layers.add(topLayer);
        layers.add(middleLayer);
        network.setLayers(layers);

        assertEquals(firstConnection, network.getConnection(topLayer, middleLayer));
        assertEquals(firstConnection, network.getConnection(middleLayer, topLayer));
    }

    @Test
    public void returnsNullWhenThereIsntAConnectingLayer() {
        Set<Layer> layers = new HashSet<Layer>();
        Layer topLayer = new Layer();
        Layer middleLayer = new Layer();
        Layer bottomLayer = new Layer();

        Connections firstConnection = new FullyConnected(topLayer, middleLayer, 0, 0);

        topLayer.addConnection(firstConnection);

        layers.add(topLayer);
        layers.add(middleLayer);
        network.setLayers(layers);

        assertNull(network.getConnection(topLayer, bottomLayer));
        assertNull(network.getConnection(bottomLayer, topLayer));
    }

    @Test
    public void addLayerDoesntAddNullLayer() {
        assertFalse(network.addLayer(null));
    }

    @Test
    public void addLayerAddsUniqueLayer() {
        Layer topLayer = new Layer();

        assertTrue(network.addLayer(topLayer));
        assertEquals(topLayer, network.getOutputLayer());
    }

    @Test
    public void addLayerToANullLayerList() {
        Layer topLayer = new Layer();

        network.setLayers(null);
        assertTrue(network.addLayer(topLayer));
        assertEquals(topLayer, network.getOutputLayer());
    }

    @Test
    public void addLayerRejectsDuplicateLayer() {
        Layer topLayer = new Layer();

        assertTrue(network.addLayer(topLayer));
        assertFalse(network.addLayer(topLayer));
        assertEquals(topLayer, network.getOutputLayer());
    }

    @Test
    public void removeLayerDoesntRemoveNullLayer() {
        assertFalse(network.removeLayer(null));
    }

    @Test
    public void removesTopLayer() {
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

        assertTrue(network.removeLayer(topLayer));
    }

    @Test
    public void removesTopAndMiddleLayer() {
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

        assertTrue(network.removeLayer(topLayer));
        assertTrue(network.removeLayer(middleLayer));
    }

    @Test
    public void removesNonexistentLayer() {
        Set<Layer> layers = new HashSet<Layer>();
        Layer topLayer = new Layer();
        Layer middleLayer = new Layer();
        Layer bottomLayer = new Layer();

        layers.add(topLayer);
        layers.add(middleLayer);
        layers.add(bottomLayer);
        network.setLayers(layers);

        assertTrue(network.removeLayer(new Layer()));
    }

    @Test
    public void removeLayerFromNullLayerList() {
        network.setLayers(null);
        assertFalse(network.removeLayer(new Layer()));
    }

    @Test
    public void addLayersWithANullListDoesntRemoveOldLayers() {
        List<Layer> layers = new ArrayList<Layer>();
        layers.add(new Layer());
        layers.add(new Layer());
        layers.add(new Layer());

        NeuralNetworkImpl network = new NeuralNetworkImpl(layers);

        network.addLayers(null);
        assertEquals(3, network.getLayers().size());
    }

    @Test
    public void addLayersAddsUniqueLayers() {
        List<Layer> layers = new ArrayList<Layer>();
        layers.add(new Layer());
        layers.add(new Layer());
        layers.add(new Layer());

        NeuralNetworkImpl network = new NeuralNetworkImpl(layers);

        layers = new ArrayList<Layer>();
        layers.add(new Layer());
        layers.add(new Layer());

        network.addLayers(layers);
        assertEquals(5, network.getLayers().size());
    }

    @Test
    public void addLayerDoesntAddExistingLayer() {
        List<Layer> layers = new ArrayList<Layer>();
        Layer oldLayer = new Layer();
        layers.add(oldLayer);
        layers.add(new Layer());
        layers.add(new Layer());

        NeuralNetworkImpl network = new NeuralNetworkImpl(layers);

        layers = new ArrayList<Layer>();
        layers.add(oldLayer);
        layers.add(new Layer());

        network.addLayers(layers);
        assertEquals(4, network.getLayers().size());
    }

    @Test
    public void addLayersAddsLayersToNullLayerList() {
        network.setLayers(null);

        List<Layer> layers = new ArrayList<Layer>();
        layers.add(new Layer());
        layers.add(new Layer());

        network.addLayers(layers);
        assertEquals(2, network.getLayers().size());
    }

    @Test
    public void addConnectionDoesntAddNullConnection() {
        network.addConnection(null);
        assertEquals(0, network.getConnections().size());
    }

    @Test
    public void addConnectionAddsNewConnection() {
        Layer topLayer = new Layer();
        Layer middleLayer = new Layer();

        Connections connection = new FullyConnected(topLayer, middleLayer, 0, 0);

        network.addConnection(connection);
        assertEquals(1, network.getConnections().size());
    }
}
