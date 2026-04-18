import 'package:flutter/material.dart';
import 'screens/onboarding_screen.dart';
import 'screens/ai_lens_screen.dart';
import 'theme/tana_theme.dart';

void main() {
  runApp(const TanaApp());
}

class TanaApp extends StatelessWidget {
  const TanaApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'TAना Heritage Explorer',
      debugShowCheckedModeBanner: false,
      theme: TanaTheme.lightTheme,
      initialRoute: '/',
      routes: {
        '/': (context) => const OnboardingScreen(),
        '/lens': (context) => const AILensScreen(),
      },
    );
  }
}
