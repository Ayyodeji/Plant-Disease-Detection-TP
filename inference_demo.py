"""
Inference Demo Script

Demonstrates how to use the trained models for prediction.
"""

import sys
from pathlib import Path
import argparse
import json

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from deployment import InferencePipeline


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(
        description="Plant Disease Detection - Inference Demo"
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to input image'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['keras', 'tflite', 'onnx', 'sklearn'],
        default='keras',
        help='Type of model'
    )
    parser.add_argument(
        '--class-mapping',
        type=str,
        default='data/processed/class_mapping.json',
        help='Path to class mapping JSON'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Number of top predictions to show'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("PLANT DISEASE DETECTION - INFERENCE DEMO")
    print("=" * 70)
    
    # Initialize inference pipeline
    print(f"\nLoading model: {args.model_path}")
    pipeline = InferencePipeline(
        model_path=args.model_path,
        class_mapping_path=args.class_mapping,
        model_type=args.model_type
    )
    
    # Make prediction
    print(f"\nAnalyzing image: {args.image}")
    result = pipeline.predict(
        args.image,
        top_k=args.top_k,
        return_probabilities=False
    )
    
    # Display results
    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)
    
    print(f"\nTop Prediction:")
    top_pred = result['top_prediction']
    print(f"  Disease: {top_pred['class']}")
    print(f"  Confidence: {top_pred['confidence_percentage']:.2f}%")
    
    print(f"\nTop {args.top_k} Predictions:")
    for i, pred in enumerate(result['top_k_predictions'], 1):
        print(f"  {i}. {pred['class']}: {pred['confidence_percentage']:.2f}%")
    
    print("\n" + "=" * 70)
    
    # Optionally save results
    output_path = Path(args.image).parent / f"{Path(args.image).stem}_prediction.json"
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()


