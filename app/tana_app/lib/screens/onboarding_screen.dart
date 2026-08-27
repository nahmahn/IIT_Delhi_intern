import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:flutter_animate/flutter_animate.dart';

class OnboardingScreen extends StatelessWidget {
  const OnboardingScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    
    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          // Background Image with Ken Burns Effect
          Image.asset(
            'assets/images/onboarding_bg.png',
            fit: BoxFit.cover,
          )
          .animate(onPlay: (controller) => controller.repeat(reverse: true))
          .scale(
            duration: 10.seconds,
            begin: const Offset(1.0, 1.0),
            end: const Offset(1.15, 1.15),
            curve: Curves.easeInOutQuad,
          ),
          
          // Gradient Overlay
          Container(
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: [
                  Colors.black.withValues(alpha: 0.3),
                  Colors.black.withValues(alpha: 0.9),
                ],
              ),
            ),
          ),
          
          // Main Content
          SafeArea(
            child: LayoutBuilder(
              builder: (context, constraints) {
                return SingleChildScrollView(
                  child: ConstrainedBox(
                    constraints: BoxConstraints(
                      minHeight: constraints.maxHeight,
                    ),
                    child: Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 32.0, vertical: 48.0),
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          // Top Branding
                          Column(
                            children: [
                              const SizedBox(height: 24),
                              Text(
                                'TAना',
                                style: GoogleFonts.notoSerif(
                                  fontSize: 72,
                                  fontWeight: FontWeight.bold,
                                  fontStyle: FontStyle.italic,
                                  color: Colors.white,
                                  letterSpacing: -2,
                                ),
                              )
                              .animate()
                              .fadeIn(duration: 800.ms)
                              .slideY(begin: -0.2, end: 0, curve: Curves.easeOutQuad),
                              
                              const SizedBox(height: 8),
                              Container(
                                height: 1.5,
                                width: 120,
                                color: theme.colorScheme.primary.withValues(alpha: 0.6),
                              )
                              .animate()
                              .scaleX(duration: 1000.ms, delay: 400.ms, curve: Curves.easeOutBack),
                            ],
                          ),
                          
                          const SizedBox(height: 32),
                          
                          // Central Text
                          Column(
                            children: [
                              Text(
                                'Identifying and Preserving Indian Textile Heritage through AI.',
                                textAlign: TextAlign.center,
                                style: GoogleFonts.notoSerif(
                                  fontSize: 28,
                                  height: 1.2,
                                  color: Colors.white,
                                ),
                              )
                              .animate()
                              .fadeIn(duration: 800.ms, delay: 600.ms)
                              .slideY(begin: 0.2, end: 0, curve: Curves.easeOutQuad),
                              
                              const SizedBox(height: 16),
                              Text(
                                'भारतीय वस्त्र विरासत की पहचान।',
                                textAlign: TextAlign.center,
                                style: GoogleFonts.notoSerif(
                                  fontSize: 18,
                                  fontStyle: FontStyle.italic,
                                  color: theme.colorScheme.surfaceContainerHighest.withValues(alpha: 0.9),
                                ),
                              )
                              .animate()
                              .fadeIn(duration: 800.ms, delay: 900.ms),
                              
                              const SizedBox(height: 40),
                              
                              // Feature Pills (Wrapped to prevent horizontal overflow)
                              Wrap(
                                alignment: WrapAlignment.center,
                                spacing: 12,
                                runSpacing: 12,
                                children: [
                                  _buildPill('AI VISION • दृष्टि', theme)
                                      .animate()
                                      .fadeIn(delay: 1200.ms)
                                      .scale(curve: Curves.easeOutBack),
                                  _buildPill('ARCHIVE • संग्रह', theme)
                                      .animate()
                                      .fadeIn(delay: 1400.ms)
                                      .scale(curve: Curves.easeOutBack),
                                ],
                              ),
                            ],
                          ),
                          
                          const SizedBox(height: 32),
                  
                  // Action Button
                  Column(
                    children: [
                      ElevatedButton(
                        onPressed: () => Navigator.pushReplacementNamed(context, '/lens'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: theme.colorScheme.primary,
                          foregroundColor: theme.colorScheme.onPrimary,
                          padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 24),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(16),
                          ),
                          elevation: 12,
                        ),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Text(
                              'GET STARTED • प्रारंभ',
                              style: theme.textTheme.labelLarge?.copyWith(
                                fontSize: 16,
                                letterSpacing: 2.5,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                            const SizedBox(width: 8),
                            const Icon(Icons.chevron_right, size: 24),
                          ],
                        ),
                      )
                      .animate(onPlay: (controller) => controller.repeat(reverse: true))
                      .shimmer(duration: 2500.ms, color: Colors.white.withValues(alpha: 0.3))
                      .animate() // Separate chain for entry
                      .fadeIn(delay: 1800.ms)
                      .slideY(begin: 0.5, end: 0, curve: Curves.easeOutBack),
                      
                      const SizedBox(height: 48),
                    ],
                  ),
                        ],
                      ),
                    ),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPill(String label, ThemeData theme) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: theme.colorScheme.surfaceContainerHighest.withValues(alpha: 0.3)),
        color: theme.colorScheme.surfaceContainerHighest.withValues(alpha: 0.1),
      ),
      child: Text(
        label,
        style: GoogleFonts.inter(
          fontSize: 10,
          fontWeight: FontWeight.bold,
          letterSpacing: 1.5,
          color: Colors.white,
        ),
      ),
    );
  }
}
