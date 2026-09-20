package me.cortex.voxy.client;

import me.cortex.voxy.client.core.IGetVoxyRenderSystem;
import me.cortex.voxy.client.core.VoxyRenderSystem;
import me.cortex.voxy.client.core.gl.Capabilities;
import me.cortex.voxy.client.core.gpu.RenderBackend;
import me.cortex.voxy.client.core.gpu.RenderBackendFactory;
import me.cortex.voxy.client.core.model.bakery.BudgetBufferRenderer;
import me.cortex.voxy.client.core.rendering.util.SharedIndexBuffer;
import me.cortex.voxy.common.Logger;
import me.cortex.voxy.commonImpl.VoxyCommon;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandRegistrationCallback;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.ChatFormatting;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.components.debug.DebugScreenDisplayer;
import net.minecraft.client.gui.components.debug.DebugScreenEntries;
import net.minecraft.client.gui.components.debug.DebugScreenEntry;
import net.minecraft.resources.Identifier;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.chunk.LevelChunk;
import org.jspecify.annotations.Nullable;

import java.util.HashSet;
import java.util.function.Consumer;
import java.util.function.Function;

public class VoxyClient implements ClientModInitializer {
    private static final HashSet<String> FREX = new HashSet<>();

    /** How often {@code VOXY_DEV_TIME} re-issues {@code time set} to hold the clock still. */
    private static final int REPIN_TICKS = 100;

    /** An integer env var, or the default when unset, blank or unparseable. */
    private static int parseEnvIntDefault(final String name, final int def) {
        final String v = System.getenv(name);
        if (v == null || v.isBlank()) return def;
        try {
            return Integer.parseInt(v.trim());
        } catch (NumberFormatException e) {
            Logger.error(name + " must be an integer; got \"" + v + "\"", e);
            return def;
        }
    }

    private static long gcd(long a, long b) {
        while (b != 0) {
            long t = a % b;
            a = b;
            b = t;
        }
        return a == 0 ? 1 : a;
    }

    public static void initVoxyClient() {
        Capabilities.init();//Ensure clinit is called

        if (Capabilities.INSTANCE.hasBrokenDepthSampler) {
            Logger.error("AMD broken depth sampler detected, voxy does not work correctly and has been disabled, this will hopefully be fixed in the future");
        }

        // Query the active render backend instead of OpenGL capabilities directly:
        // on macOS Apple Silicon the OpenGL driver tops out at 4.1 (no compute,
        // no indirect draws), which would fail this check even though the Metal
        // backend supports both. MetalRenderBackend reports compute and indirect
        // as available; the OpenGL backend delegates to the legacy Capabilities.
        RenderBackend backend = RenderBackendFactory.get();
        Logger.info("Render backend: " + backend.getType()
                + " (compute=" + backend.hasCompute()
                + ", indirectParameters=" + backend.hasIndirectParameters() + ")");

        boolean systemSupported = backend.hasCompute() && backend.hasIndirectParameters() && !Capabilities.INSTANCE.hasBrokenDepthSampler;

        // M9 transitional: even though MetalRenderBackend reports compute=true and
        // indirectParameters=true, Voxy's render path (MDICSectionRenderer,
        // HiZBuffer2, HierarchicalOcclusionTraverser, ChunkBoundRenderer,
        // bakery, AbstractRenderPipeline) is still raw OpenGL DSA and aborts
        // the JVM the first time the GL driver hits an unsupported call on
        // Apple's frozen GL 4.1. Until that work lands (M9-M11 migration to
        // the encoder API + IOSurface bridge), force-disable Voxy on
        // non-OpenGL backends so the mixins find VoxyRenderSystem == null
        // and Sodium's chunk render runs unmodified — the user sees normal
        // close-distance MC + Sodium rendering instead of a blue screen.
        // Feature flag to opt into the Metal render path. The flag exists
        // separately from regular env vars so opting in is intentional —
        // until the full MDIC encoder migration lands, this path produces
        // a clear color via the IOSurfaceBridge, not actual LOD chunks.
        boolean forceMetal = "1".equals(System.getenv("VOXY_FORCE_METAL"))
                || "true".equals(System.getenv("VOXY_FORCE_METAL"))
                || "true".equals(System.getProperty("voxy.forceMetal", "false"));

        if (systemSupported && backend.getType() != me.cortex.voxy.client.core.gpu.BackendType.OPENGL) {
            if (forceMetal) {
                Logger.info("[VOXY_FORCE_METAL] Voxy enabled on " + backend.getType()
                        + " backend. Under Metallum the LOD pass renders straight into the "
                        + "frame's colour and depth attachments; there is no bridge and no "
                        + "composite.");
            } else {
                Logger.warn("[M9 TRANSITIONAL] Voxy disabled on " + backend.getType()
                        + " backend. Set VOXY_FORCE_METAL=1 to enable the Metal render path.");
                systemSupported = false;
            }
        }

        if (systemSupported) {

            if (backend.getType() == me.cortex.voxy.client.core.gpu.BackendType.OPENGL) {
                // Both of these are raw-GL startup: id() creates the GL index buffer and
                // BudgetBufferRenderer compiles a GL program. Under whole-frame Metal there is no
                // GL context at all, and glCreateShader does not fail gracefully -- LWJGL aborts
                // the JVM ("FATAL ERROR in native method: No context is current"). The Metal paths
                // allocate their own resources lazily, and MetalBudgetBufferRenderer is dormant.
                SharedIndexBuffer.INSTANCE.id();
                BudgetBufferRenderer.init();
            }

            VoxyCommon.setInstanceFactory(VoxyClientInstance::new);

            if (!Capabilities.INSTANCE.subgroup) {
                Logger.warn("GPU does not support subgroup operations, expect some performance degradation");
            }

        } else {
            Logger.error("Voxy is unsupported on your system.");
        }
    }

    @Override
    public void onInitializeClient() {
        // Iris-pack + Metal coexistence now happens at the SOLID-pass head:
        // IrisGbufferInjector draws the LOD bridge into the pack's terrain
        // gbuffer (MixinDefaultChunkRenderer). The former HUD-time late
        // composite that lived here is gone — it ran after Iris finalized but
        // painted over the pack's post chain and skipped entirely with the
        // HUD hidden (F1).
        DebugScreenEntries.register(Identifier.fromNamespaceAndPath("voxy", "version"), new DebugScreenEntry() {
            @Override
            public void display(DebugScreenDisplayer lines, @Nullable Level level, @Nullable LevelChunk levelChunk, @Nullable LevelChunk levelChunk2) {
                if (!VoxyCommon.isAvailable()) {
                    lines.addLine(ChatFormatting.RED + "voxy-"+VoxyCommon.MOD_VERSION);//Voxy installed, not avalible
                    return;
                }
                var instance = VoxyCommon.getInstance();
                if (instance == null) {
                    lines.addLine(ChatFormatting.YELLOW + "voxy-" + VoxyCommon.MOD_VERSION);//Voxy avalible, no instance active
                    return;
                }
                VoxyRenderSystem vrs = null;
                var wr = Minecraft.getInstance().levelRenderer;
                if (wr != null) vrs = ((IGetVoxyRenderSystem) wr).getVoxyRenderSystem();

                //Voxy instance active
                lines.addLine((vrs==null?ChatFormatting.DARK_GREEN:ChatFormatting.GREEN)+"voxy-"+VoxyCommon.MOD_VERSION);
            }
        });

        DebugScreenEntries.register(Identifier.fromNamespaceAndPath("voxy","debug"), new VoxyDebugScreenEntry());
        ClientCommandRegistrationCallback.EVENT.register((dispatcher, registryAccess) -> {
            if (VoxyCommon.isAvailable()) {
                dispatcher.register(VoxyCommands.register());
            }
        });

        // VOXY_AUTO_SCREENSHOT=<seconds>: periodically save an in-game
        // screenshot via MC's own Screenshot API (lands in run/screenshots).
        // Debug-loop aid — headless verification can capture frames without
        // desktop screencapture (which fails when other windows are
        // frontmost on the test machine).
        String autoShot = System.getenv("VOXY_AUTO_SCREENSHOT");
        if (autoShot != null && !autoShot.isBlank()) {
            int parsedInterval;
            try {
                parsedInterval = Math.max(2, Integer.parseInt(autoShot.trim()));
            } catch (NumberFormatException e) {
                parsedInterval = 10;
            }
            final long intervalNanos = parsedInterval * 1_000_000_000L;
            final long[] last = {System.nanoTime()};
            net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents.END_CLIENT_TICK.register(client -> {
                if (client.level == null || client.gameRenderer.mainRenderTarget() == null) return;
                long now = System.nanoTime();
                if (now - last[0] < intervalNanos) return;
                last[0] = now;
                net.minecraft.client.Screenshot.grab(client.gameDirectory,
                        client.gameRenderer.mainRenderTarget(), component -> {});
            });
            Logger.info("VOXY_AUTO_SCREENSHOT active: every " + parsedInterval + "s");
        }

        // VOXY_DEV_TIME=<time>: pin the world clock a few seconds after joining, so screenshots are
        // comparable. Debug-loop aid.
        //
        // Without this, brightness comparisons are meaningless: the day/night cycle runs at 20 real
        // minutes per in-game day, so two runs an hour apart land at arbitrary times and "the LOD
        // renders black" is indistinguishable from "it is night". A black frame with a bright hotbar
        // is exactly the night signature, and that ambiguity cost a round of diagnosis.
        //
        // Accepts any /time set argument ("noon", "midnight", "day", or a tick count), and also
        // clears the weather, which darkens the sky the same way.
        //
        // The clock is held by RE-ISSUING `time set`, not by `gamerule doDaylightCycle false`.
        // That gamerule call does not work on 26.2 — the server answers "Incorrect argument for
        // command / gamerule doDaylightCycle false<--[HERE]", i.e. the rule name parses and the
        // boolean value does not — and `doWeatherCycle` fails identically. Both failures were
        // silent to everything except the log: `time set noon` and `weather clear` DO succeed, so
        // the run looked pinned and was not. A day is 24000 ticks over 20 real minutes, so the
        // clock walks out of the pinned value at 20 ticks/second, and a four-minute capture drifts
        // 4800 ticks — from noon (6000) to mid-afternoon (10800). Every frame in such a run was
        // lit from a different sun angle while the log said "pinned", which is the same class of
        // error as the camera fall this class already documents: an uncontrolled variable that
        // reads as a property of the build under test.
        //
        // Re-issuing the command is immune to the gamerule API change and costs one command per
        // REPIN_TICKS. It also self-corrects if a later `/time add` or a sleeping player moves it.
        // VOXY_DEV_TIME_HOLD=0 restores the pre-fix behaviour — set the clock once and let the
        // daylight cycle walk it away — so "held" and "drifting" can be compared as one runtime
        // switch on one SHA rather than as two builds. A rebuild between arms is the confound this
        // project keeps paying for; a switch is not.
        final boolean holdClock = !"0".equals(System.getenv("VOXY_DEV_TIME_HOLD"));
        String devTime = System.getenv("VOXY_DEV_TIME");
        if (devTime != null && !devTime.isBlank()) {
            final String timeArg = devTime.trim();
            final int[] ticksUntilApply = {40};   // let the world finish loading first
            final int[] repinCountdown = {0};
            final boolean[] applied = {false};
            net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents.END_CLIENT_TICK.register(client -> {
                if (client.level == null) return;
                var server = client.getSingleplayerServer();
                if (server == null) return;
                try {
                    // withSuppressedOutput: `time set` is re-issued every REPIN_TICKS, and a
                    // command source that reports prints "[Server: Set minecraft:overworld to
                    // time marker minecraft:noon]" into chat — two lines, every five seconds,
                    // across the middle of every screenshot. Same failure as the `tp @s`
                    // feedback this file already documents: the harness covering the exact
                    // pixels under test. `sendCommandFeedback false` does not suppress a
                    // command's own output; only the source's output flag does.
                    var source = server.createCommandSourceStack().withSuppressedOutput();
                    var commands = server.getCommands();
                    if (!applied[0]) {
                        if (ticksUntilApply[0]-- > 0) return;
                        applied[0] = true;
                        for (String cmd : new String[] {
                                "gamerule doDaylightCycle false",
                                "gamerule doWeatherCycle false",
                                "weather clear",
                                "time set " + timeArg}) {
                            commands.performPrefixedCommand(source, cmd);
                        }
                        repinCountdown[0] = REPIN_TICKS;
                        Logger.info("VOXY_DEV_TIME active: world clock "
                                + (holdClock
                                        ? "held at '" + timeArg + "' by re-issuing time set every "
                                                + REPIN_TICKS + " ticks (the doDaylightCycle gamerule"
                                                + " does not apply on 26.2)"
                                        : "set to '" + timeArg + "' ONCE and left to drift"
                                                + " (VOXY_DEV_TIME_HOLD=0)"));
                        return;
                    }
                    if (!holdClock) return;
                    if (repinCountdown[0]-- > 0) return;
                    repinCountdown[0] = REPIN_TICKS;
                    commands.performPrefixedCommand(source, "time set " + timeArg);
                } catch (Throwable t) {
                    Logger.error("VOXY_DEV_TIME failed to apply '" + timeArg + "'", t);
                    applied[0] = true;
                }
            });
        }

        // VOXY_DEV_CAM="x,y,z[,yaw,pitch]": pin the camera for the whole run, so two builds produce
        // comparable frames. Debug-loop aid, and the fix for the reason visual A/B has kept lying.
        //
        // VOXY_DEV_TIME is not enough. It pins the clock, but the test world spawns the player in the
        // air, so the camera is FALLING: every screenshot is taken at a different height and pitch
        // even though the x/z read the same. Measured this session -- the same build reported the
        // black fraction as 11.5% and the sky fraction as 6.7%, then 10.4%/18.3%, then 26.1%/18.5%
        // while the camera x/z never moved off (246,-235). Every one of those differences is the fall,
        // not the change under test, and a candidate fix was nearly accepted and another nearly
        // rejected on that basis. Spectator mode plus a periodic tp holds the eye still; the optional
        // yaw/pitch are omitted from the tp when absent, which keeps the spawn orientation (identical
        // across runs of the same world) instead of inventing one.
        String devCam = System.getenv("VOXY_DEV_CAM");
        if (devCam != null && !devCam.isBlank()) {
            final String[] parts = devCam.trim().split("\\s*,\\s*");
            final String tpArgs;
            if (parts.length >= 5) {
                tpArgs = parts[0] + " " + parts[1] + " " + parts[2] + " " + parts[3] + " " + parts[4];
            } else if (parts.length == 3) {
                tpArgs = parts[0] + " " + parts[1] + " " + parts[2];
            } else {
                throw new IllegalStateException(
                        "VOXY_DEV_CAM must be \"x,y,z\" or \"x,y,z,yaw,pitch\"; got \"" + devCam + "\"");
            }
            final int[] ticksUntilApply = {40};
            final boolean[] applied = {false};
            final int[] reapplyCountdown = {20};
            // VOXY_DEV_CAM_DRIFT=<blocks per second>: translate the pinned camera continuously instead
            // of holding it still. Added because the splotches are reported to come back as soon as
            // the camera moves at all, and a pinned camera cannot reproduce that -- but a *rotating*
            // one breaks every screenshot comparison, since the sky band changes and the catcher uses
            // it to decide which frames may be compared. Translation only: the sky is a uniform
            // gradient, so it stays pixel-identical while the terrain moves underneath, which keeps
            // the capture metric valid while the camera is genuinely in motion.
            final double drift;
            {
                String d = System.getenv("VOXY_DEV_CAM_DRIFT");
                double parsed = 0;
                if (d != null && !d.isBlank()) {
                    try {
                        parsed = Double.parseDouble(d.trim());
                    } catch (NumberFormatException e) {
                        Logger.error("VOXY_DEV_CAM_DRIFT must be a number of blocks per second", e);
                    }
                }
                drift = parsed;
            }
            final double baseX;
            try {
                baseX = Double.parseDouble(parts[0]);
            } catch (NumberFormatException e) {
                throw new IllegalStateException("VOXY_DEV_CAM x must be a number; got \"" + parts[0] + "\"", e);
            }
            final String tail = parts.length >= 5
                    ? " " + parts[3] + " " + parts[4]
                    : "";
            final int[] driftTicks = {0};
            // VOXY_DEV_CAM_SPIN=<degrees per second>: rotate the pinned camera's yaw continuously while
            // its POSITION stays fixed. The third movement instrument, added because the first two are
            // both measurably wrong for bug 3 in opposite directions:
            //
            //   VOXY_DEV_TP       reproduces nothing. 227 frames over 112 arrivals, median 714
            //                     near-black px, nothing above 983 except the world-load overlay. The
            //                     artefact needs CONTINUOUS motion and a teleport is one jump.
            //   VOXY_DEV_CAM_DRIFT (translation) does reproduce, but only after the camera has flown
            //                     far enough to leave the populated world: its counts ramp monotonically
            //                     (+4,300 px per 2 s) and the frames show the LOD fragmenting into
            //                     detached slabs -- which is confounded with "flew out of the data" and
            //                     so cannot be attributed to the bug. At 6 blocks/s it is ~1,700 blocks
            //                     out by the time it fires.
            //
            // Yaw rotation is the user's own description of the trigger -- "it also shows up when I move
            // my camera too fast" -- and it moves the camera not at all, so the world stays populated
            // and nothing is confounded. The reason it was avoided before is that rotation breaks the
            // sky-band camera check the screenshot catcher used to decide which frames may be compared;
            // that check existed to make pairwise DIFFING valid, and the metric in use now is a
            // per-frame absolute near-black census, which needs no comparability at all.
            final double spin;
            {
                String s = System.getenv("VOXY_DEV_CAM_SPIN");
                double parsed = 0;
                if (s != null && !s.isBlank()) {
                    try {
                        parsed = Double.parseDouble(s.trim());
                    } catch (NumberFormatException e) {
                        Logger.error("VOXY_DEV_CAM_SPIN must be a number of degrees per second", e);
                    }
                }
                spin = parsed;
            }
            final String baseYaw = parts.length >= 4 ? parts[3] : "0";
            final String basePitch = parts.length >= 5 ? parts[4] : "0";
            if (drift != 0) {
                Logger.info("VOXY_DEV_CAM_DRIFT active: translating " + drift + " blocks/sec, no rotation");
            }
            if (spin != 0) {
                // A constant spin ALIASES against a constant capture interval, and the aliasing is
                // silent: the camera advances spin*interval degrees between screenshots, so the
                // headings actually sampled are 360/gcd(advance,360) of them, not all of them. At the
                // first rate tried -- 60 deg/s with the 2 s capture -- the advance was exactly 120 deg,
                // gcd(120,360)=120, so EVERY frame came from one of THREE headings while looking like
                // a full survey. A direction-dependent artefact would then be either always or never
                // seen depending on phase, and the run would read as a clean result.
                //
                // The interval is re-parsed here rather than shared, because the capture registry is
                // configured in a different block above and a shared local would be a silent coupling.
                // Unset means no capture at all; the arithmetic below is then about a run that will not
                // produce frames anyway, so the default only has to keep the numbers finite.
                final int capSecs = Math.max(2, parseEnvIntDefault("VOXY_AUTO_SCREENSHOT", 2));
                final long advance = Math.round(Math.abs(spin) * capSecs);
                long a = advance % 360;
                if (a == 0) a = 360;                       // a full turn samples the same heading forever
                final long headings = 360 / gcd(a, 360);
                Logger.info("VOXY_DEV_CAM_SPIN active: rotating " + spin + " deg/sec at a fixed position"
                        + " -- " + advance + " deg per " + capSecs + "s capture, "
                        + headings + " distinct heading(s) sampled");
                if (headings < 30) {
                    Logger.warn("VOXY_DEV_CAM_SPIN is ALIASED against the " + capSecs + "s capture: only "
                            + headings + " distinct heading(s) will ever be screenshotted, so this run "
                            + "samples " + headings + " direction(s) rather than the whole circle. Pick a "
                            + "rate whose advance is not a large divisor of 360 (e.g. 67 deg/s at 2s "
                            + "gives 134 deg, 180 headings).");
                }
            }
            net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents.END_CLIENT_TICK.register(client -> {
                if (client.level == null) return;
                var server = client.getSingleplayerServer();
                if (server == null) return;
                // The command source needs BOTH the player (so `@s` resolves) and level-4 permission
                // (so tp/gamemode are allowed). The server console stack has the permission but no
                // entity, so `@s` would fail and the command would report failure to nobody; the
                // player's own stack has the entity but no op. Take the console stack and attach the
                // player to it.
                //
                // withSuppressedOutput, not `gamerule sendCommandFeedback false`: that gamerule does
                // not suppress a player's own command output, so the periodic re-tp printed
                // "Teleported ..." five lines deep over the middle of every screenshot, covering the
                // exact pixels under test.
                var players = server.getPlayerList().getPlayers();
                if (players.isEmpty()) return;
                var source = server.createCommandSourceStack().withEntity(players.get(0))
                        .withSuppressedOutput();
                var commands = server.getCommands();
                try {
                    if (!applied[0]) {
                        if (ticksUntilApply[0]-- > 0) return;
                        applied[0] = true;
                        // Spectator: no gravity, no collision, and the LOD ring still follows the eye.
                        commands.performPrefixedCommand(source, "gamemode spectator");
                        commands.performPrefixedCommand(source, "tp @s " + tpArgs);
                        Logger.info("VOXY_DEV_CAM active: pinned the camera to " + tpArgs);
                        return;
                    }
                    // Re-assert periodically: the first tp can land before the world finishes loading
                    // the chunks under it, and MC will nudge a player that ends up inside geometry.
                    // Under drift it re-asserts every tick, which is what makes the motion continuous
                    // rather than a one-block step per second.
                    if (drift != 0 || spin != 0) {
                        driftTicks[0]++;
                        if (reapplyCountdown[0]-- > 0) return;
                        reapplyCountdown[0] = 1;
                        final String args;
                        if (spin != 0) {
                            // Yaw only: position and pitch are re-asserted unchanged, so the eye never
                            // moves and the world under it stays resident.
                            args = parts[0] + " " + parts[1] + " " + parts[2] + " "
                                    + String.format(java.util.Locale.ROOT, "%.4f",
                                            (Double.parseDouble(baseYaw) + spin * (driftTicks[0] / 20.0)) % 360.0)
                                    + " " + basePitch;
                        } else {
                            String x = String.format(java.util.Locale.ROOT, "%.4f",
                                    baseX + drift * (driftTicks[0] / 20.0));
                            args = x + " " + parts[1] + " " + parts[2] + tail;
                        }
                        commands.performPrefixedCommand(source, "tp @s " + args);
                        return;
                    }
                    if (reapplyCountdown[0]-- > 0) return;
                    reapplyCountdown[0] = 20;
                    commands.performPrefixedCommand(source, "tp @s " + tpArgs);
                } catch (Throwable t) {
                    Logger.error("VOXY_DEV_CAM failed to pin the camera to " + tpArgs, t);
                    applied[0] = true;
                }
            });
        }


        //
        // VOXY_DEV_TP="<p1>;<p2>[;...]" where each point is "x,y,z" or "x,y,z,yaw,pitch": teleport the
        // eye between two fixed points on a cycle and screenshot the FIRST frame at each arrival.
        //
        // Movement is the trigger for the black splotches, and a pinned camera cannot reproduce it --
        // but free movement makes every frame incomparable, because the camera differs between any two
        // captures. A teleport is the one form of movement that is both real and repeatable: it forces
        // the full re-traversal and re-anchor that a moving camera does, yet it lands on a known
        // position, so every "arrival at point A" frame shows the same scene and can be compared with
        // every other. Give both points the same yaw/pitch and the sky band is identical too, which
        // keeps the screenshot catcher's camera gate satisfied rather than splitting every frame into
        // its own group.
        //
        // VOXY_DEV_TP_TICKS (default 40) is how long to hold at each point; VOXY_DEV_TP_SHOT_DELAY
        // (default 2) is how many ticks after the teleport to grab, since the client's position and
        // the rendered frame both lag the command by a tick.
        String devTp = System.getenv("VOXY_DEV_TP");        if (devTp != null && !devTp.isBlank()) {
            final String[] points = devTp.trim().split("\\s*;\\s*");
            if (points.length < 2) {
                throw new IllegalStateException("VOXY_DEV_TP needs at least two ';'-separated points");
            }
            for (String p : points) {
                int n = p.split("\\s*,\\s*").length;
                if (n != 3 && n != 5) {
                    throw new IllegalStateException(
                            "VOXY_DEV_TP point must be \"x,y,z\" or \"x,y,z,yaw,pitch\"; got \"" + p + "\"");
                }
            }
            final int holdTicks = Math.max(4, parseEnvIntDefault("VOXY_DEV_TP_TICKS", 40));
            final int shotDelay = Math.max(1, parseEnvIntDefault("VOXY_DEV_TP_SHOT_DELAY", 2));
            // 1 is the old behaviour: a single grab, `shotDelay` ticks after each arrival.
            final int shotBurst = Math.max(1, parseEnvIntDefault("VOXY_DEV_TP_SHOT_BURST", 1));
            final boolean takeShots = !"0".equals(System.getenv("VOXY_DEV_TP_SHOT"));
            Logger.info("VOXY_DEV_TP active: cycling " + points.length + " point(s) every "
                    + holdTicks + " ticks, screenshot " + shotDelay + " tick(s) after each arrival"
                    + (shotBurst > 1 ? ", burst of " + shotBurst + " ticks" : ""));

            final int[] ticksUntilApply = {40};
            final int[] hold = {0};
            final int[] shotIn = {-1};
            final int[] burstLeft = {0};
            final int[] idx = {0};
            final boolean[] applied = {false};
            net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents.END_CLIENT_TICK.register(client -> {
                if (client.level == null) return;
                var server = client.getSingleplayerServer();
                if (server == null) return;
                var players = server.getPlayerList().getPlayers();
                if (players.isEmpty()) return;
                var source = server.createCommandSourceStack().withEntity(players.get(0))
                        .withSuppressedOutput();
                var commands = server.getCommands();
                try {
                    if (!applied[0]) {
                        if (ticksUntilApply[0]-- > 0) return;
                        applied[0] = true;
                        commands.performPrefixedCommand(source, "gamemode spectator");
                    }
                    // The grab lands a fixed number of ticks after the teleport, so it is the first
                    // frame the new eye position could have produced.
                    //
                    // VOXY_DEV_TP_SHOT_BURST=<n> grabs once per TICK for n ticks from `shotDelay`
                    // onwards, instead of a single frame, because one sample cannot catch a transient
                    // that lasts one to three frames. It must be one grab per tick, not n grabs in one
                    // tick: Screenshot.grab reads the current render target asynchronously, so n calls
                    // in a tick capture the same framebuffer n times over.
                    //
                    // Reason to expect a transient exactly here: a teleport forces the full
                    // re-traversal and re-anchor, and the measured artefact counts ramp immediately
                    // after a discontinuity -- so the frames just after arrival are where a
                    // short-lived fault should be, and the default single grab at +2 ticks samples
                    // that window once rather than surveying it.
                    if (shotIn[0] > 0 && --shotIn[0] == 0) {
                        burstLeft[0] = shotBurst;
                    }
                    if (burstLeft[0] > 0 && takeShots
                            && client.gameRenderer.mainRenderTarget() != null) {
                        burstLeft[0]--;
                        final int pi = (idx[0] + points.length - 1) % points.length;
                        Logger.info("[Metal-TP-SHOT] point=" + pi
                                + " burst=" + (shotBurst - 1 - burstLeft[0])
                                + " args=" + points[pi]);
                        net.minecraft.client.Screenshot.grab(client.gameDirectory,
                                client.gameRenderer.mainRenderTarget(), component -> {});
                    }
                    if (hold[0]-- > 0) return;
                    hold[0] = holdTicks;
                    final String pos = points[idx[0] % points.length];
                    idx[0]++;
                    commands.performPrefixedCommand(source, "tp @s " + pos);
                    Logger.info("[Metal-TP] -> point " + ((idx[0] - 1) % points.length) + " args=" + pos);
                    shotIn[0] = shotDelay;
                    burstLeft[0] = 0;   // an unfinished burst must not bleed into the next arrival
                } catch (Throwable t) {
                    Logger.error("VOXY_DEV_TP failed", t);
                    applied[0] = true;
                }
            });
        }

        //
        // NOTE: tagging screen-open intervals here would let the capture tools exclude pause-menu
        // frames, which are a large near-black frame and therefore indistinguishable to any pixel
        // metric from the artefact under investigation. It is not implemented: 26.2's `Minecraft` has
        // no `Screen`-typed field or accessor in this mapping, so there is nothing to read. Menu frames
        // are currently excluded by hand, from the log's own shutdown markers.
        //

        //
        // Voxy's frame-rate diagnostic (Metal-RING) only exists when Voxy's render system is
        // running, so it cannot measure the vanilla-only baseline of an A/B -- which is exactly the
        // measurement that decides whether a slowdown belongs to Voxy or to what it renders into.
        // This reads MC's own counter, so it reports either way.
        String fpsLog = System.getenv("VOXY_FPS_LOG");
        if (fpsLog != null && !fpsLog.isBlank()) {
            final long periodNanos;
            try {
                periodNanos = Math.max(1, Long.parseLong(fpsLog.trim())) * 1_000_000_000L;
            } catch (NumberFormatException e) {
                throw new IllegalStateException("VOXY_FPS_LOG must be an integer number of seconds", e);
            }
            final long[] last = {System.nanoTime()};
            net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents.END_CLIENT_TICK.register(client -> {
                long now = System.nanoTime();
                if (now - last[0] < periodNanos) return;
                last[0] = now;
                Logger.info("[FPS] " + client.getFps()
                        + (client.level == null ? "  (no world)" : "  chunks=" + client.level.getChunkSource().getLoadedChunksCount()));
            });
        }

        // VOXY_TEST_CAMERA="x,y,z,yaw,pitch": pin the player's position and facing, in flight,
        // re-applied every tick so server corrections cannot drift the view.
        //
        // Position is what makes a capture comparable at all. Two frames of the same build — or of
        // two builds — can only be diffed if the camera is identical, and the alternative (issuing
        // /tp) is unreliable: Minecraft rewrites level.dat's allowCommands on save, so a world
        // patched to allow cheats silently reverts and every command comes back
        // "Unknown or incomplete command". Driving the entity directly sidesteps world settings
        // entirely, and works on a server too.
        String testCamera = System.getenv("VOXY_TEST_CAMERA");
        if (testCamera != null && !testCamera.isBlank()) {
            String[] parts = testCamera.split(",");
            if (parts.length != 5) {
                throw new IllegalStateException(
                        "VOXY_TEST_CAMERA needs \"x,y,z,yaw,pitch\"; got: " + testCamera);
            }
            final double cx = Double.parseDouble(parts[0].trim());
            final double cy = Double.parseDouble(parts[1].trim());
            final double cz = Double.parseDouble(parts[2].trim());
            final float cyaw = Float.parseFloat(parts[3].trim());
            final float cpitch = Float.parseFloat(parts[4].trim());
            net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents.END_CLIENT_TICK.register(client -> {
                if (client.player == null) return;
                var p = client.player;
                p.setPos(cx, cy, cz);
                p.setYRot(cyaw);
                p.setXRot(cpitch);
                p.setDeltaMovement(0, 0, 0);
                p.resetFallDistance();
                var ab = p.getAbilities();
                ab.mayfly = true;
                ab.flying = true;
            });
            Logger.info("VOXY_TEST_CAMERA active: " + testCamera + " (flying, re-pinned each tick)");
        }

        // VOXY_JOIN_COMMAND="/time set noon; /weather clear; /tp 0 120 0 0 30":
        // run slash-commands once a few seconds after joining a world, then stop.
        //
        // This is what makes a capture *comparable*. Comparing two builds — or two
        // runs of one build — only means something if both rendered the same scene,
        // and time of day, weather, position and facing are otherwise whatever the
        // save happened to be left holding. Pairs with VOXY_AUTO_SCREENSHOT; needs
        // allowCommands on the world (a test fixture, not a gameplay world).
        String joinCommand = System.getenv("VOXY_JOIN_COMMAND");
        if (joinCommand != null && !joinCommand.isBlank()) {
            final int delayTicks;
            try {
                delayTicks = Math.max(20, Integer.parseInt(
                        System.getenv().getOrDefault("VOXY_JOIN_COMMAND_DELAY", "60").trim()));
            } catch (NumberFormatException e) {
                throw new IllegalStateException("VOXY_JOIN_COMMAND_DELAY must be an integer", e);
            }
            final String[] commands = joinCommand.split(";");
            final int[] ticks = {0};
            final boolean[] done = {false};
            net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents.END_CLIENT_TICK.register(client -> {
                if (done[0] || client.level == null || client.player == null
                        || client.getConnection() == null) {
                    return;
                }
                if (++ticks[0] < delayTicks) return;
                done[0] = true;
                for (String raw : commands) {
                    String cmd = raw.trim();
                    if (cmd.isEmpty()) continue;
                    // sendCommand takes the body without the leading slash.
                    client.getConnection().sendCommand(cmd.startsWith("/") ? cmd.substring(1) : cmd);
                }
                Logger.info("VOXY_JOIN_COMMAND ran after " + delayTicks + " ticks: " + joinCommand);
            });
        }

        FabricLoader.getInstance()
                .getEntrypoints("frex_flawless_frames", Consumer.class)
                .forEach(api -> ((Consumer<Function<String,Consumer<Boolean>>>)api).accept(name->active->{if (active) {
                    FREX.add(name);
                } else {
                    FREX.remove(name);
                }}));
    }

    public static boolean isFrexActive() {
        return !FREX.isEmpty();
    }

    public static int getOcclusionDebugState() {
        return 0;
    }

    public static boolean disableSodiumChunkRender() {
        return false;// getOcclusionDebugState() != 0;
    }
}
