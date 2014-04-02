package com.github.neuralnetworks.architecture.test;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.List;

import org.junit.Before;
import org.junit.Test;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.FullyConnected;
import com.github.neuralnetworks.architecture.Layer;

public class LayerTest {

    private Layer layer;

    @Before
    public void setup() {
        layer = new Layer();
    }

    @Test
    public void testNewLayerHasEmptyConnectionsList() {
        assertEquals(0, layer.getConnections().size());
    }

    @Test
    public void testAddAConnection() {
        layer.addConnection(new FullyConnected(new Layer(), new Layer(), 0, 0));

        assertEquals(1, layer.getConnections().size());
    }

    @Test
    public void testAddAListOfConnection() {
        List<Connections> connectionList = new ArrayList<Connections>();

        connectionList.add(new FullyConnected(new Layer(), new Layer(), 0, 0));
        connectionList.add(new FullyConnected(new Layer(), new Layer(), 0, 0));
        connectionList.add(new FullyConnected(new Layer(), new Layer(), 0, 0));
        connectionList.add(new FullyConnected(new Layer(), new Layer(), 0, 0));

        layer.setConnections(connectionList);

        assertEquals(4, layer.getConnections().size());
    }

}
