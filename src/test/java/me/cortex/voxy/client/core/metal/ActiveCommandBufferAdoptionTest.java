package me.cortex.voxy.client.core.metal;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Pins the command-buffer adoption rule that took several bring-up iterations to get right.
 *
 * <p>Metal's {@code _status < MTLCommandBufferStatusCommitted} assertion carries no Java stack and
 * aborts the process, so the failure mode here is a hard crash with nothing pointing at the cause.
 * The rule itself is one comparison, which makes it exactly the kind of thing worth holding still:
 * getting it wrong is expensive to rediscover, and cheap to pin.
 */
class ActiveCommandBufferAdoptionTest {

    private static final long METALLUM_BUFFER = 0x852e78a80L;
    private static final long STALE_BUFFER = 0x852e78000L;

    @Test
    void adoptsWhenCachedBufferIsStale() {
        // Metallum committed the frame Voxy was encoding into and opened a new one. Holding the old
        // handle means encoding into a committed buffer, which Metal aborts on.
        assertTrue(MetalRenderBackend.shouldAdoptMetallumBuffer(STALE_BUFFER, METALLUM_BUFFER));
    }

    @Test
    void doesNotAdoptWhenAlreadyOnMetallumsBuffer() {
        // Re-adopting the same handle is harmless but would also drop a live blit batch for nothing.
        assertFalse(MetalRenderBackend.shouldAdoptMetallumBuffer(METALLUM_BUFFER, METALLUM_BUFFER));
    }

    @Test
    void adoptsWhenVoxyHasNoBufferYet() {
        assertTrue(MetalRenderBackend.shouldAdoptMetallumBuffer(0L, METALLUM_BUFFER));
    }

    @Test
    void doesNotAdoptWhenMetallumHasNoBuffer() {
        // Metallum absent or between frames: there is nothing to adopt, and callers fall back to
        // creating their own buffer rather than dropping the one they hold.
        assertFalse(MetalRenderBackend.shouldAdoptMetallumBuffer(STALE_BUFFER, 0L));
        assertFalse(MetalRenderBackend.shouldAdoptMetallumBuffer(0L, 0L));
    }
}
