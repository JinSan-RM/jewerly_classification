from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import timm
import io
import base64
from datetime import datetime
import uuid

app = Flask(__name__)
CORS(app)

# 업로드 및 결과 폴더 생성
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ConvNeXt 기반 모델 클래스 (제공된 코드와 동일)
class MemoryEfficientClassifier(nn.Module):
    def __init__(self, num_classes=5, model_size='xlarge'):
        super(MemoryEfficientClassifier, self).__init__()

        model_name = 'convnext_base.fb_in22k_ft_in1k'
        
        try:
            self.backbone = timm.create_model(model_name, pretrained=True)
            backbone_features = self.backbone.num_features
            
            if hasattr(self.backbone, 'head'):
                self.backbone.head = nn.Identity()
            elif hasattr(self.backbone, 'classifier'):
                self.backbone.classifier = nn.Identity()
            else:
                for name, module in self.backbone.named_modules():
                    if isinstance(module, nn.Linear) and module.out_features > num_classes:
                        setattr(self.backbone, name.split('.')[-1], nn.Identity())
                        break
            
            print(f"ConvNeXt 백본 로드: {model_name}")
            print(f"백본 특성 차원: {backbone_features}")
            
        except Exception as e:
            print(f"ConvNeXt 모델 로드 실패: {str(e)}")
            raise RuntimeError(f"ConvNeXt 모델을 로드할 수 없습니다: {str(e)}")

        hidden_size = 1024

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(backbone_features, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        if hasattr(self.backbone, 'forward_features'):
            features = self.backbone.forward_features(x)
        else:
            features = self.backbone(x)
        
        output = self.classifier(features)
        return output

# 전역 변수
model = None
device = None
transform = None
class_names = ['anklet', 'bracelet', 'earring', 'necklace', 'ring']

def get_test_transform(image_size=320):
    """이미지 전처리 변환"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_model_safely(model_path, num_classes=5, device_name='cpu'):
    """모델 로드"""
    print(f"모델 로딩 중 경로: {model_path}")
    
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=device_name)
        
        if isinstance(checkpoint, dict) and not hasattr(checkpoint, 'eval'):
            print("state_dict 형태로 저장된 모델입니다.")
            model = MemoryEfficientClassifier(num_classes=num_classes)
            
            try:
                model.load_state_dict(checkpoint, strict=True)
                print("state_dict 로드 성공 (strict=True)")
            except RuntimeError as e:
                print(f"strict 로드 실패: {str(e)}")
                print("strict=False로 재시도")
                model.load_state_dict(checkpoint, strict=False)
                print("state_dict 로드 성공 (strict=False)")
        else:
            print("전체 모델로 저장되었습니다.")
            model = checkpoint
        
        model = model.to(device_name)
        model.eval()
        
        print("모델 로드 및 설정 완료!")
        return model
        
    except Exception as e:
        print(f"모델 로드 실패: {str(e)}")
        return None

def predict_single_image_with_tta(model, image, transform, device, num_tta=5):
    """TTA를 적용한 이미지 예측"""
    try:
        # PIL Image를 텐서로 변환
        image_tensor = transform(image).unsqueeze(0).to(device)

        model.eval()
        predictions = []

        with torch.no_grad():
            # 원본 이미지 예측
            with autocast():
                outputs = model(image_tensor)
                predictions.append(F.softmax(outputs, dim=1))

            # TTA 변환들
            tta_transforms = [
                lambda x: torch.flip(x, dims=[3]),  # 수평 반전
                lambda x: torch.flip(x, dims=[2]),  # 수직 반전
                lambda x: torch.rot90(x, k=2, dims=[2, 3]),  # 180도 회전
                lambda x: x * 0.95 + 0.02,  # 밝기 조정
            ]

            # TTA 적용
            for i, tta_transform in enumerate(tta_transforms[:num_tta-1]):
                try:
                    transformed = tta_transform(image_tensor.clone())
                    with autocast():
                        outputs = model(transformed)
                        predictions.append(F.softmax(outputs, dim=1))
                except:
                    continue

            # 평균 예측
            if predictions:
                avg_prediction = torch.mean(torch.stack(predictions), dim=0)
                class_probabilities = avg_prediction[0].cpu().numpy()
                return class_probabilities

    except Exception as e:
        print(f"예측 오류: {str(e)}")

    return None

def init_model():
    """모델 초기화"""
    global model, device, transform
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 장치: {device}")
    
    # 모델 로드
    model_path = 'jewerly_classification_AI_model.pth'  # 실제 모델 파일 경로로 수정
    model = load_model_safely(model_path, len(class_names), device)
    
    if model is None:
        print("모델 로드 실패")
        return False
    
    # 이미지 전처리 변환 설정
    transform = get_test_transform(320)
    
    print("모델 초기화 완료")
    return True

@app.route('/')
def index():
    """메인 페이지"""
    return send_from_directory('.', 'main.html')

@app.route('/predict', methods=['POST'])
def predict():
    """이미지 분류 예측 API"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '파일이 업로드되지 않았습니다.'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': '파일이 선택되지 않았습니다.'})
        
        # 이미지 로드 및 변환
        image = Image.open(file.stream).convert('RGB')
        
        # 원본 이미지 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{uuid.uuid4().hex[:8]}_{file.filename}"
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        image.save(image_path)
        
        # 모델 예측 수행
        if model is None:
            return jsonify({'success': False, 'error': '모델이 로드되지 않았습니다.'})
        
        # TTA 예측 수행
        probabilities = predict_single_image_with_tta(model, image, transform, device, num_tta=5)
        
        if probabilities is None:
            return jsonify({'success': False, 'error': '예측 중 오류가 발생했습니다.'})
        
        # 결과 정리
        predictions = []
        for i, prob in enumerate(probabilities):
            predictions.append({
                'category': class_names[i],
                'probability': float(prob)
            })
        
        # 확률순으로 정렬
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        # 최상위 예측 결과
        top_prediction = predictions[0]
        
        # 결과 저장
        result_data = {
            'timestamp': timestamp,
            'filename': filename,
            'original_filename': file.filename,
            'predictions': predictions,
            'top_prediction': {
                'category': top_prediction['category'],
                'confidence': top_prediction['probability']
            }
        }
        
        # 결과를 JSON 파일로 저장
        result_filename = f"result_{timestamp}_{uuid.uuid4().hex[:8]}.json"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        
        import json
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        print(f"예측 완료: {top_prediction['category']} ({top_prediction['probability']:.3f})")
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'top_prediction': top_prediction,
            'saved_image': filename,
            'saved_result': result_filename
        })
        
    except Exception as e:
        print(f"예측 API 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'서버 오류: {str(e)}'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """업로드된 파일 서빙"""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/results/<filename>')
def result_file(filename):
    """결과 파일 서빙"""
    return send_from_directory(RESULT_FOLDER, filename)

@app.route('/health')
def health_check():
    """헬스 체크"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if device else 'unknown'
    })

if __name__ == '__main__':
    print("주얼리 분류 서버 시작 중")
    
    # 모델 초기화
    if init_model():
        print("서버 준비 완료")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("모델 초기화 실패. 서버를 시작할 수 없습니다.")