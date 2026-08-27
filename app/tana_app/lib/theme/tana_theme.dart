import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class TanaTheme {
  static const Color primaryColor = Color(0xFF914D00);
  static const Color primaryContainer = Color(0xFFF28C28);
  static const Color backgroundColor = Color(0xFFFCF9F3);
  static const Color surfaceColor = Color(0xFFFCF9F3);
  static const Color onPrimary = Color(0xFFFFFFFF);
  static const Color onBackground = Color(0xFF1C1C18);
  static const Color surfaceVariant = Color(0xFFE5E2DC);
  static const Color onSurfaceVariant = Color(0xFF554336);

  static ThemeData get lightTheme {
    return ThemeData(
      useMaterial3: true,
      colorScheme: ColorScheme.fromSeed(
        seedColor: primaryColor,
        primary: primaryColor,
        onPrimary: onPrimary,
        primaryContainer: primaryContainer,
        surface: backgroundColor,
        onSurface: onBackground,
        surfaceContainerHighest: surfaceVariant,
        onSurfaceVariant: onSurfaceVariant,
      ),
      textTheme: TextTheme(
        displayLarge: GoogleFonts.notoSerif(
          fontSize: 32,
          fontWeight: FontWeight.bold,
          color: primaryColor,
        ),
        headlineLarge: GoogleFonts.notoSerif(
          fontSize: 24,
          fontWeight: FontWeight.bold,
          color: onBackground,
        ),
        bodyLarge: GoogleFonts.inter(
          fontSize: 16,
          color: onBackground,
        ),
        bodyMedium: GoogleFonts.inter(
          fontSize: 14,
          color: onSurfaceVariant,
        ),
        labelLarge: GoogleFonts.inter(
          fontSize: 14,
          fontWeight: FontWeight.w600,
          letterSpacing: 1.2,
          color: onPrimary,
        ),
      ),
    );
  }
}
