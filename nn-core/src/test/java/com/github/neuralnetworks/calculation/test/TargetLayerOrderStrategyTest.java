package com.github.neuralnetworks.calculation.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

import java.util.HashSet;
import java.util.Set;

import org.junit.Before;
import org.junit.Test;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.calculation.TargetLayerOrderStrategy;

public class TargetLayerOrderStrategyTest {

    private NeuralNetworkImpl network;
    private TargetLayerOrderStrategy strategy;
    private Layer topLayer;
    private Layer middleLayer;
    private Layer bottomLayer;

    @Before
    public void setup() {
        network = new NeuralNetworkImpl();

        Set<Layer> layers = new HashSet<Layer>();
        topLayer = new Layer();
        middleLayer = new Layer();
        bottomLayer = new Layer();

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
    }

    @Test
    public void setNeuralNetwork() {
        strategy = new TargetLayerOrderStrategy(null, null, new HashSet<Layer>());
        assertNull(strategy.getNeuralNetwork());

        strategy.setNeuralNetwork(network);
        assertEquals(network, strategy.getNeuralNetwork());
    }

    @Test
    public void setTargetLayer() {
        strategy = new TargetLayerOrderStrategy(null, null, new HashSet<Layer>());
        assertNull(strategy.getTargetLayer());

        strategy.setTargetLayer(topLayer);
        assertEquals(topLayer, strategy.getTargetLayer());
    }

    @Test
    public void setCalculatedLayers() {
        strategy = new TargetLayerOrderStrategy(null, null, new HashSet<Layer>());
        assertEquals(0, strategy.getCalculatedLayers().size());

        HashSet<Layer> calculatedLayers = new HashSet<Layer>();
        calculatedLayers.add(new Layer());
        calculatedLayers.add(new Layer());
        calculatedLayers.add(new Layer());

        strategy.setCalculatedLayers(calculatedLayers);
        assertEquals(3, strategy.getCalculatedLayers().size());
    }

    @Test
    public void orderWithTwoLayers() {
        HashSet<Layer> calculatedLayers = new HashSet<Layer>();
        calculatedLayers.add(bottomLayer);

        strategy = new TargetLayerOrderStrategy(network, topLayer, calculatedLayers);
        assertEquals(2, strategy.order().size());
    }

    @Test
    public void orderWithOneLayer() {
        HashSet<Layer> calculatedLayers = new HashSet<Layer>();
        calculatedLayers.add(bottomLayer);

        strategy = new TargetLayerOrderStrategy(network, middleLayer, calculatedLayers);
        assertEquals(1, strategy.order().size());
    }

    @Test
    public void orderWithIncorrectTargetReturnsEmptySet() {
        HashSet<Layer> calculatedLayers = new HashSet<Layer>();
        calculatedLayers.add(bottomLayer);

        strategy = new TargetLayerOrderStrategy(network, new Layer(), calculatedLayers);
        assertEquals(0, strategy.order().size());
    }
}
