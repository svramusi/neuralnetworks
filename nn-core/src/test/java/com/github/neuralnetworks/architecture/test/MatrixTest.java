package com.github.neuralnetworks.architecture.test;

import org.junit.Test;

import com.github.neuralnetworks.architecture.Matrix;

public class MatrixTest {

    // This is strictly used to point out a defect in the code and needs to be
    // fixed
    @Test(expected = NullPointerException.class)
    public void uninitializedMatrixThrowsExceptionWhenGettingRows() {
        Matrix matrix = new Matrix();
        matrix.getRows();
    }
}
