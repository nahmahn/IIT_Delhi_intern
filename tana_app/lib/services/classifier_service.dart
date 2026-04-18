import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class ClassifierService {
  late Interpreter _yoloInterpreter;
  late Interpreter _resnetInterpreter;

  final List<String> _labels = [
    'baluchari',
    'maheshwari',
    'negammam',
    'phulkari',
  ];

  static const int _imgSize = 640;
  static const double _confidenceThreshold = 0.40;

  // Optimized weights from Python ensemble verification
  static const double _yoloWeight = 0.85;
  static const double _resnetWeight = 0.15;

  // ImageNet normalization constants for ResNet50
  static const List<double> _mean = [0.485, 0.456, 0.406];
  static const List<double> _std = [0.229, 0.224, 0.225];

  bool _isInit = false;

  Future<void> init() async {
    try {
      // Load both models in parallel
      final results = await Future.wait([
        Interpreter.fromAsset('assets/models/yolo11m_4class.tflite'),
        Interpreter.fromAsset('assets/models/resnet50_4saree.tflite'),
      ]);
      
      _yoloInterpreter = results[0];
      _resnetInterpreter = results[1];
      _isInit = true;
      debugPrint('Ensemble models loaded successfully');
    } catch (e) {
      debugPrint('Init error: $e');
    }
  }

  Future<Map<String, dynamic>> classify(File imageFile) async {
    if (!_isInit) await init();

    final bytes = await imageFile.readAsBytes();
    final image = img.decodeImage(bytes);
    if (image == null) return {'label': 'Error', 'confidence': 0.0};

    // Preprocess once for each model requirements
    final yoloInput = _prepareYoloInput(image);
    final resnetInput = _prepareResnetInput(image);

    // Prepare outputs
    final yoloOutput = List.filled(_labels.length, 0.0).reshape([1, _labels.length]);
    final resnetOutput = List.filled(_labels.length, 0.0).reshape([1, _labels.length]);

    // Run inference
    _yoloInterpreter.run(yoloInput.reshape([1, _imgSize, _imgSize, 3]), yoloOutput);
    _resnetInterpreter.run(resnetInput.reshape([1, _imgSize, _imgSize, 3]), resnetOutput);

    final yoloProbs = List<double>.from(yoloOutput[0]);
    final resnetProbs = List<double>.from(resnetOutput[0]);

    // Ensemble: Weighted Average
    final ensembleProbs = List<double>.filled(_labels.length, 0.0);
    for (int i = 0; i < _labels.length; i++) {
      ensembleProbs[i] = (yoloProbs[i] * _yoloWeight) + (resnetProbs[i] * _resnetWeight);
    }

    // Find best class
    int maxIdx = 0;
    double maxVal = -1;
    for (int i = 0; i < ensembleProbs.length; i++) {
      if (ensembleProbs[i] > maxVal) {
        maxVal = ensembleProbs[i];
        maxIdx = i;
      }
    }

    if (maxVal < _confidenceThreshold) {
      return {'label': 'unknown', 'confidence': maxVal};
    }

    return {'label': _labels[maxIdx], 'confidence': maxVal};
  }

  /// 0 to 1 scaling (Standard for YOLO)
  Float32List _prepareYoloInput(img.Image image) {
    final resized = _resizeAndCrop(image, _imgSize);
    final input = Float32List(_imgSize * _imgSize * 3);
    int pixelIndex = 0;

    for (int y = 0; y < _imgSize; y++) {
      for (int x = 0; x < _imgSize; x++) {
        final pixel = resized.getPixel(x, y);
        input[pixelIndex++] = pixel.r / 255.0;
        input[pixelIndex++] = pixel.g / 255.0;
        input[pixelIndex++] = pixel.b / 255.0;
      }
    }
    return input;
  }

  /// ImageNet Normalization (Required for ResNet50)
  Float32List _prepareResnetInput(img.Image image) {
    // Standard ResNet transform: Resize(ShortSide=672) -> CenterCrop(640)
    final resized = _resizeAndCrop(image, _imgSize, resizeShortSide: 672);
    final input = Float32List(_imgSize * _imgSize * 3);
    int pixelIndex = 0;

    for (int y = 0; y < _imgSize; y++) {
      for (int x = 0; x < _imgSize; x++) {
        final pixel = resized.getPixel(x, y);
        // (x / 255.0 - mean) / std
        input[pixelIndex++] = (pixel.r / 255.0 - _mean[0]) / _std[0];
        input[pixelIndex++] = (pixel.g / 255.0 - _mean[1]) / _std[1];
        input[pixelIndex++] = (pixel.b / 255.0 - _mean[2]) / _std[2];
      }
    }
    return input;
  }

  img.Image _resizeAndCrop(img.Image image, int size, {int? resizeShortSide}) {
    final w = image.width;
    final h = image.height;
    
    // Default: resize shortest side to 'size'
    int targetShortSide = resizeShortSide ?? size;
    double scale = targetShortSide / (w < h ? w : h);
    
    int newW = (w * scale).toInt();
    int newH = (h * scale).toInt();
    
    final scaledImage = img.copyResize(image, width: newW, height: newH, interpolation: img.Interpolation.linear);

    final left = (newW - size) ~/ 2;
    final top = (newH - size) ~/ 2;
    return img.copyCrop(scaledImage, x: left, y: top, width: size, height: size);
  }

  void dispose() {
    _yoloInterpreter.close();
    _resnetInterpreter.close();
  }
}
