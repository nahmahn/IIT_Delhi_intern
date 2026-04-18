import 'dart:io';
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:flutter_animate/flutter_animate.dart';
import '../services/classifier_service.dart';
import '../theme/tana_theme.dart';

class AILensScreen extends StatefulWidget {
  const AILensScreen({super.key});

  @override
  State<AILensScreen> createState() => _AILensScreenState();
}

class _AILensScreenState extends State<AILensScreen> {
  CameraController? _controller;
  final ClassifierService _classifier = ClassifierService();

  String _resultLabel = '';
  double _confidence = 0.0;
  bool _isProcessing = false;
  bool _showResult = false;

  List<CameraDescription>? _cameras;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    _cameras = await availableCameras();
    if (_cameras == null || _cameras!.isEmpty) return;

    _controller = CameraController(
      _cameras![0],
      ResolutionPreset.high,
      enableAudio: false,
    );

    await _controller!.initialize();
    await _classifier.init();

    if (!mounted) return;
    setState(() {});
  }

  Future<void> _captureAndClassify() async {
    if (_controller == null || !_controller!.value.isInitialized) return;
    if (_isProcessing) return;

    setState(() {
      _isProcessing = true;
      _showResult = false;
    });

    try {
      final image = await _controller!.takePicture();
      final result = await _classifier.classify(File(image.path));

      if (mounted) {
        setState(() {
          final raw = result['label'];
          _resultLabel = raw[0].toUpperCase() + raw.substring(1);
          _confidence = result['confidence'];
          _isProcessing = false;
          _showResult = true;
        });
      }
    } catch (e) {
      setState(() {
        _resultLabel = "Error Identifying";
        _isProcessing = false;
        _showResult = true;
      });
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_controller == null || !_controller!.value.isInitialized) {
      return const Scaffold(
        backgroundColor: Colors.black,
        body: Center(child: CircularProgressIndicator(color: TanaTheme.primaryColor)),
      );
    }

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          // Camera Preview
          CameraPreview(_controller!),

          // Scanning Animation (only when processing)
          if (_isProcessing) _buildScanningLine(),

          // Decorative Viewfinder
          _buildViewfinder(),

          // Top Header
          _buildTopBar(),

          // Result Overlay (Glassmorphism)
          if (_showResult) _buildResultCard(),

          // Camera Controls
          Positioned(
            bottom: 120,
            left: 0,
            right: 0,
            child: _buildCaptureButton(),
          ),

          // Bottom Navigation
          Positioned(
            bottom: 0,
            left: 0,
            right: 0,
            child: _buildBottomNav(),
          ),
        ],
      ),
    );
  }

  Widget _buildScanningLine() {
    return Center(
      child: SizedBox(
        width: 300,
        height: 300,
        child: Stack(
          children: [
            Positioned(
              top: 0,
              left: 0,
              right: 0,
              child: Container(
                height: 2,
                decoration: BoxDecoration(
                  boxShadow: [
                    BoxShadow(
                      color: TanaTheme.primaryColor.withValues(alpha: 0.8),
                      blurRadius: 10,
                      spreadRadius: 2,
                    ),
                  ],
                  color: TanaTheme.primaryColor,
                ),
              ),
            ).animate(onPlay: (controller) => controller.repeat())
             .moveY(begin: 0, end: 300, duration: 1500.ms, curve: Curves.easeInOut),
          ],
        ),
      ),
    );
  }

  Widget _buildResultCard() {
    return Positioned(
      bottom: 220,
      left: 24,
      right: 24,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(24),
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
          child: Container(
            padding: const EdgeInsets.all(24),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(24),
              border: Border.all(color: Colors.white.withValues(alpha: 0.2)),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(8),
                      decoration: BoxDecoration(
                        color: TanaTheme.primaryColor.withValues(alpha: 0.2),
                        shape: BoxShape.circle,
                      ),
                      child: const Icon(Icons.auto_awesome, color: TanaTheme.primaryColor, size: 20),
                    ),
                    const SizedBox(width: 12),
                    Text(
                      'IDENTIFIED HERITAGE',
                      style: GoogleFonts.inter(
                        fontSize: 12,
                        fontWeight: FontWeight.w900,
                        color: Colors.white70,
                        letterSpacing: 2,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                Text(
                  _resultLabel,
                  style: GoogleFonts.notoSerif(
                    fontSize: 32,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    Expanded(
                      child: LinearProgressIndicator(
                        value: _confidence,
                        backgroundColor: Colors.white10,
                        color: TanaTheme.primaryColor,
                        borderRadius: BorderRadius.circular(4),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Text(
                      '${(_confidence * 100).toStringAsFixed(1)}%',
                      style: GoogleFonts.inter(color: Colors.white60, fontWeight: FontWeight.bold),
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                GestureDetector(
                  onTap: () => setState(() => _showResult = false),
                  child: Container(
                    padding: const EdgeInsets.symmetric(vertical: 12),
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: Colors.white24),
                    ),
                    child: const Center(
                      child: Text('DISMISS', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold, letterSpacing: 1.5)),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    ).animate().fadeIn(duration: 400.ms).slideY(begin: 0.2, end: 0, curve: Curves.easeOutBack);
  }

  Widget _buildCaptureButton() {
    return Center(
      child: GestureDetector(
        onTap: _captureAndClassify,
        child: Stack(
          alignment: Alignment.center,
          children: [
            // Breathing Glow
            Container(
              width: 90,
              height: 90,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                border: Border.all(color: TanaTheme.primaryColor.withValues(alpha: 0.5), width: 2),
              ),
            ).animate(onPlay: (controller) => controller.repeat(reverse: true))
             .scale(begin: const Offset(1, 1), end: const Offset(1.2, 1.2), duration: 2.seconds, curve: Curves.easeInOut)
             .fadeOut(begin: 0.5, duration: 2.seconds),

            // Inner Button
            Container(
              width: 70,
              height: 70,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: _isProcessing ? Colors.grey[800] : Colors.white,
                boxShadow: [
                  BoxShadow(color: Colors.white.withValues(alpha: 0.2), blurRadius: 20, spreadRadius: 5),
                ],
              ),
              child: _isProcessing 
                ? const Center(child: SizedBox(width: 24, height: 24, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white)))
                : const Icon(Icons.camera_alt, color: Colors.black, size: 32),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildViewfinder() {
    return Center(
      child: Container(
        width: 300,
        height: 300,
        decoration: BoxDecoration(
          border: Border.all(color: Colors.white24, width: 1.5),
          borderRadius: BorderRadius.circular(40),
        ),
        child: Stack(
          children: [
            _viewfinderCorner(top: 0, left: 0, rot: 0),
            _viewfinderCorner(top: 0, right: 0, rot: 90),
            _viewfinderCorner(bottom: 0, left: 0, rot: 270),
            _viewfinderCorner(bottom: 0, right: 0, rot: 180),
          ],
        ),
      ).animate(onPlay: (controller) => controller.repeat(reverse: true))
       .scale(begin: const Offset(1.0, 1.0), end: const Offset(1.02, 1.02), duration: 3.seconds),
    );
  }

  Widget _viewfinderCorner({double? top, double? bottom, double? left, double? right, required double rot}) {
    return Positioned(
      top: top, bottom: bottom, left: left, right: right,
      child: Transform.rotate(
        angle: rot * 0.0174533,
        child: Container(
          width: 40,
          height: 40,
          decoration: const BoxDecoration(
            border: Border(
              top: BorderSide(color: TanaTheme.primaryColor, width: 4),
              left: BorderSide(color: TanaTheme.primaryColor, width: 4),
            ),
            borderRadius: BorderRadius.only(topLeft: Radius.circular(12)),
          ),
        ),
      ),
    );
  }

  Widget _buildTopBar() {
    return Positioned(
      top: 0, left: 0, right: 0,
      child: Container(
        padding: const EdgeInsets.fromLTRB(24, 64, 24, 32),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [Colors.black.withValues(alpha: 0.8), Colors.transparent],
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
          ),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            IconButton(
              icon: const Icon(Icons.arrow_back_ios, color: Colors.white),
              onPressed: () => Navigator.pop(context),
            ),
            Text(
              'HERITAGE LENS',
              style: GoogleFonts.notoSerif(
                fontSize: 16,
                fontWeight: FontWeight.w900,
                color: TanaTheme.primaryColor,
                letterSpacing: 3,
              ),
            ),
            const Icon(Icons.flash_off, color: Colors.white54),
          ],
        ),
      ),
    );
  }

  Widget _buildBottomNav() {
    return ClipRRect(
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
        child: Container(
          padding: const EdgeInsets.fromLTRB(24, 20, 24, 40),
          decoration: BoxDecoration(
            color: Colors.black.withValues(alpha: 0.4),
            borderRadius: const BorderRadius.vertical(top: Radius.circular(32)),
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              _nav(Icons.dashboard_rounded, false),
              _nav(Icons.center_focus_strong, true),
              _nav(Icons.history_edu_rounded, false),
              _nav(Icons.person_outline_rounded, false),
            ],
          ),
        ),
      ),
    );
  }

  Widget _nav(IconData icon, bool active) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: active ? BoxDecoration(
        color: TanaTheme.primaryColor.withValues(alpha: 0.2),
        borderRadius: BorderRadius.circular(16),
      ) : null,
      child: Icon(
        icon,
        color: active ? TanaTheme.primaryColor : Colors.white54,
        size: 28,
      ),
    );
  }
}