package me.cortex.voxy.client.core.metal;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Metallum is not on the test classpath, so this pins the degradation contract: every accessor must
 * return a benign value and nothing may throw. Voxy runs standalone (smoke tests, non-Metal hosts)
 * far more often than it runs inside Metallum, so a hard failure here would be a real regression.
 */
class MetallumBridgeTest {

    @Test
    void reportsUnavailableWhenMetallumAbsent() {
        assertFalse(MetallumBridge.available(), "Metallum must not be detected in this classpath");
    }

    @Test
    void allHandleAccessorsDegradeToZero() {
        assertEquals(0L, MetallumBridge.device());
        assertEquals(0L, MetallumBridge.commandQueue());
        assertEquals(0L, MetallumBridge.commandBuffer());
        assertEquals(0L, MetallumBridge.renderEncoder());
        assertEquals(0L, MetallumBridge.colorAttachment());
        assertEquals(0L, MetallumBridge.depthAttachment());
        assertEquals(0, MetallumBridge.viewportWidth());
        assertEquals(0, MetallumBridge.viewportHeight());
    }

    @Test
    void endCurrentEncoderIsNoOpNotThrow() {
        assertDoesNotThrow(MetallumBridge::endCurrentEncoder);
    }

    /**
     * The reporting variants are diagnostics, but they sit on the hot path immediately before every
     * encoder creation — a throw there would turn a logging aid into the crash.
     */
    @Test
    void encoderReportingAccessorsDegradeToZero() {
        assertEquals(0L, MetallumBridge.openEncoderHandle());
        assertEquals(0L, MetallumBridge.endCurrentEncoderAndReport());
    }

    @Test
    void midFrameSplitDegradesWhenMetallumAbsent() {
        assertFalse(MetallumBridge.supportsFlushFrame());
        assertDoesNotThrow(MetallumBridge::flushFrame);
    }
}
