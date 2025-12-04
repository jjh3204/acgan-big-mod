import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import os

def convert_onnx_to_tflite(onnx_path, tflite_path):
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX file not found at {onnx_path}")
        return
    
    print(f">>> Loading ONNX model: {onnx_path}")
    onnx_model = onnx.load(onnx_path)

    # 1. ONNX -> TensorFlow SavedModel 변환
    tf_model_path = "tf_model_saved_model"

    print(">>> Converting ONNX to TensorFlow SavedModel...")
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(tf_model_path)

    # 2. TFLite Converter 설정
    print(">>> Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    
    # 최적화 옵션 설정
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # FP16 양자화 (Float16) 설정
    # 모바일 GPU(NPU) 가속에 유리하며, 모델 크기를 50% 줄임
    # converter.target_spec.supported_types = [tf.float16]          # 이 줄을 제거하면 INT8 양자화
    
    # 3. 변환 수행
    tflite_model = converter.convert()

    # 4. 파일 저장
    os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f">>> Conversion Complete! Saved to: {tflite_path}")

if __name__ == "__main__":
    deploy_dir = "./deploy"
    input_onnx = os.path.join(deploy_dir, "fruit_acgan_big.onnx")
    output_tflite = os.path.join(deploy_dir, "fruit_acgan_big_int8.tflite")

    convert_onnx_to_tflite(input_onnx, output_tflite)