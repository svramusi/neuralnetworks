package com.github.neuralnetworks.calculation.test;

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
import com.github.neuralnetworks.calculation.LayerOrderStrategy.ConnectionCandidate;
import com.github.neuralnetworks.calculation.ValuesProvider;
import com.github.neuralnetworks.calculation.neuronfunctions.ConstantConnectionCalculator;
import com.github.neuralnetworks.events.PropagationEvent;
import com.github.neuralnetworks.events.PropagationEventListener;

public class LayerCalculatorImplTest {

    private LayerCalculatorImpl layerCalc;
    ValuesProvider results;
    NeuralNetworkImpl network;
    Layer topLayer;
    Layer middleLayer;
    Layer bottomLayer;
    HashSet<Layer> calculatedLayers;

    @Before
    public void setup() {
        layerCalc = new LayerCalculatorImpl();

        results = new ValuesProvider();
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

        calculatedLayers = new HashSet<Layer>();
        calculatedLayers.add(bottomLayer);

        List<ConnectionCandidate> connections = new ArrayList<ConnectionCandidate>();
        connections.add(new ConnectionCandidate(firstConnection, middleLayer));
    }

    @Test
    public void getConnectionCalculatorReturnsNullOnUnknownLayer() {
        assertNull(layerCalc.getConnectionCalculator(new Layer()));
    }

    @Test
    public void addConnectionCalculator() {
        Layer layer = new Layer();
        ConstantConnectionCalculator calc = new ConstantConnectionCalculator();

        layerCalc.addConnectionCalculator(layer, calc);
        assertEquals(calc, layerCalc.getConnectionCalculator(layer));
    }

    @Test
    public void canAddThenRemoveConnectionCalculator() {
        Layer layer = new Layer();
        ConstantConnectionCalculator calc = new ConstantConnectionCalculator();

        layerCalc.addConnectionCalculator(layer, calc);
        assertEquals(calc, layerCalc.getConnectionCalculator(layer));

        layerCalc.removeConnectionCalculator(layer);

        assertNull(layerCalc.getConnectionCalculator(layer));
    }

    private class TestPropagationEventListener implements PropagationEventListener {
        private static final long serialVersionUID = 1L;
        private boolean handelEvent = false;

        @Override
        public void handleEvent(PropagationEvent event) {
            handelEvent = true;
        }

        public boolean wasEventHandled() {
            return handelEvent;
        }
    }

    @Test
    public void addAndRemoveSingleEventListener() {
        TestPropagationEventListener listener = new TestPropagationEventListener();

        layerCalc.addEventListener(listener);
        layerCalc.removeEventListener(listener);
    }

    @Test
    public void addAndRemoveMultipleEventListeners() {
        TestPropagationEventListener listener = new TestPropagationEventListener();
        TestPropagationEventListener listener2 = new TestPropagationEventListener();

        layerCalc.addEventListener(listener);
        layerCalc.addEventListener(listener2);
        layerCalc.removeEventListener(listener2);
        layerCalc.removeEventListener(listener);
        layerCalc.removeEventListener(listener);
    }

    @Test
    public void removeEventListenerFromEmptyList() {
        TestPropagationEventListener listener = new TestPropagationEventListener();

        layerCalc.removeEventListener(listener);
    }

    @Test
    public void calculateTriggersEvent() {
        TestPropagationEventListener listener = new TestPropagationEventListener();
        layerCalc.addEventListener(listener);

        layerCalc.calculate(network, topLayer, calculatedLayers, results);

        assertTrue(listener.wasEventHandled());
    }

    @Test
    public void calculateDoesntTriggerEventWithInvalidConnections() {
        TestPropagationEventListener listener = new TestPropagationEventListener();
        layerCalc.addEventListener(listener);

        layerCalc.calculate(network, bottomLayer, calculatedLayers, results);

        assertFalse(listener.wasEventHandled());
    }

    @Test
    public void calculateDoesntTriggerEventWhenThereIsntAListener() {
        layerCalc.calculate(network, topLayer, calculatedLayers, results);
    }
}
