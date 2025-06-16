import os
import requests
import json
import tkinter as tk
from tkinter import simpledialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageFilter
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
from dotenv import load_dotenv
import logging
from retry import retry
import multiprocessing
from rembg import new_session, remove

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY", "")
cse_id = os.getenv("GOOGLE_CSE_ID", "")
search_url = "https://www.googleapis.com/customsearch/v1"

model_files = {
    "prototxt": r"C:\Users\xuanh\.cursor-tutor\models\deploy.prototxt",
    "caffemodel": r"C:\Users\xuanh\.cursor-tutor\models\res10_300x300_ssd_iter_140000.caffemodel"
}

def get_unique_filename(folder, filename):
    base, ext = os.path.splitext(filename)

    # 🔹 품번에 '/'가 있으면 경로로 인식되지 않도록 따옴표로 감싸기
    safe_filename = f'"{base}"{ext}' if "/" in base else f"{base}{ext}"

    counter = 1
    unique_filename = safe_filename
    while os.path.exists(os.path.join(folder, unique_filename)):
        unique_filename = f'"{base}_{counter}"{ext}' if "/" in base else f"{base}_{counter}{ext}"
        counter += 1
    return unique_filename


def download_model_file(url, path):
    if not os.path.exists(path):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(response.content)
            logging.debug(f"Downloaded model file to {path}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download model file {url}: {e}")
            raise

def remove_faces_dnn(image):
    try:
        # 직접 지정한 모델 파일 경로 사용
        prototxt_path = r"C:\Users\xuanh\.cursor-tutor\models\deploy.prototxt"
        caffemodel_path = r"C:\Users\xuanh\.cursor-tutor\models\res10_300x300_ssd_iter_140000.caffemodel"

        # 모델 파일이 존재하는지 확인
        if not os.path.exists(prototxt_path) or not os.path.exists(caffemodel_path):
            raise FileNotFoundError("모델 파일이 존재하지 않습니다. 경로를 확인하세요.")

        # OpenCV DNN 모델 로드
        net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        h, w = cv_image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv_image, 1.0, (300, 300), [104, 117, 123], False, False)
        net.setInput(blob)
        detections = net.forward()

        max_face_y = 0  # 얼굴의 y축 최댓값 저장용

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # 신뢰도를 0.6으로 조정하여 더 정확한 얼굴만 탐지
            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                face_width = x2 - x1
                face_height = y2 - y1

                # 🔹 손(잘못 감지된 작은 얼굴) 필터링: 너무 작으면 무시
                if face_width < 50 or face_height < 50:
                    continue  

                # 🔹 얼굴 비율이 너무 좁거나 길면 무시 (손 가능성 높음)
                aspect_ratio = face_width / face_height
                if aspect_ratio < 0.5:
                    continue  

                # 가장 아래쪽 얼굴의 y값 저장
                if y2 > max_face_y:
                    max_face_y = y2

        # 얼굴이 감지되었을 경우, 해당 y 좌표부터 10px 위까지 전체를 흰색으로 덮기
        if max_face_y > 0:
            erase_y = max(0, max_face_y - 10)
            cv_image[:erase_y, :] = (255, 255, 255)

        return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

    except Exception as e:
        logging.error(f"Face removal failed: {e}")
        return image



def is_mostly_white(image, threshold=220, ratio=0.9):
    gray = image.convert("L")
    np_img = np.array(gray)
    white_pixels = np.sum(np_img > threshold)
    total_pixels = np_img.size
    return (white_pixels / total_pixels) > ratio

def remove_background(image):
    try:
        if is_mostly_white(image):
            logging.debug("이미지가 거의 흰색으로 구성되어 있어 배경제거 생략")
            return image

        session = new_session("u2net")
        image_bytes = BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes = image_bytes.getvalue()

        output = remove(
            image_bytes,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=215,
            alpha_matting_background_threshold=200,
            alpha_matting_erode_size=2,
            alpha_matting_base_size=2000
        )

        image_with_alpha = Image.open(BytesIO(output)).convert("RGBA")
        np_img = np.array(image_with_alpha)

        # 윤곽선 개선: bilateral filter → 부드럽고 자연스러운 경계
# 윤곽선 개선: GaussianBlur + Morphology 조합
# 윤곽선 개선: Morphology + 작은 블러 적용
        # 윤곽선 개선: GaussianBlur + Morphology 조합
        alpha = np_img[:, :, 3]

        # 1️⃣ 작은 구멍을 채워서 윤곽선 정리 (닫힘 연산)
        kernel = np.ones((3, 3), np.uint8)
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)

        # 2️⃣ GaussianBlur로 부드러운 경계 생성 (7x7 커널 사용)
        alpha = cv2.GaussianBlur(alpha, (7, 7), 0)

        # 3️⃣ 너무 뭉개진 부분 복구 (열림 연산)
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)

        # 최종 알파 채널 적용
        np_img[:, :, 3] = alpha
        processed_image = Image.fromarray(np_img)

        # 흰색 배경으로 합성
        bg = Image.new("RGBA", processed_image.size, (255, 255, 255, 255))
        bg.paste(processed_image, mask=processed_image.split()[3])

        return bg.convert("RGB")

    except Exception as e:
        logging.error(f"Background removal failed: {e}")
        return image




def get_product_names():
    input_window = tk.Tk()
    input_window.title("제품명 입력")
    input_window.geometry("400x300")
    input_window.eval('tk::PlaceWindow . center')
    
    label = tk.Label(input_window, text="제품명을 줄바꿈으로 입력하거나 파일 업로드하세요 (최대 100개):")
    label.pack(pady=10)

    text_area = ScrolledText(input_window, wrap=tk.WORD, width=50, height=10)
    text_area.pack(padx=10, pady=10)

    def upload_file():
        file_path = askopenfilename(filetypes=[("Text files", "*.txt *.csv"), ("All files", "*.*")])
        if file_path:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                text_area.delete("1.0", tk.END)
                text_area.insert(tk.END, content)

    upload_button = tk.Button(input_window, text="파일 업로드", command=upload_file)
    upload_button.pack(pady=5)

    def on_submit():
        global product_names_result
        product_names_result = text_area.get("1.0", tk.END).strip()
        input_window.destroy()

    submit_button = tk.Button(input_window, text="입력 완료", command=on_submit)
    submit_button.pack(pady=10)

    input_window.mainloop()
    return [name.strip() for name in product_names_result.splitlines() if name.strip()][:100]

def get_resolution():
    resolution_window = tk.Tk()
    resolution_window.title("해상도 입력")
    resolution_window.geometry("350x200")
    resolution_window.eval('tk::PlaceWindow . center')

    label = tk.Label(resolution_window, text="원하는 해상도를 입력하세요 (가로x세로, 예: 600x600):")
    label.pack(pady=10)

    width_frame = tk.Frame(resolution_window)
    width_frame.pack(pady=5)
    width_label = tk.Label(width_frame, text="가로 (px):")
    width_label.pack(side=tk.LEFT, padx=5)
    width_entry = tk.Entry(width_frame, width=10)
    width_entry.pack(side=tk.LEFT, padx=5)

    height_frame = tk.Frame(resolution_window)
    height_frame.pack(pady=5)
    height_label = tk.Label(height_frame, text="세로 (px):")
    height_label.pack(side=tk.LEFT, padx=5)
    height_entry = tk.Entry(height_frame, width=10)
    height_entry.pack(side=tk.LEFT, padx=5)

    def on_submit():
        global resolution
        try:
            w = int(width_entry.get())
            h = int(height_entry.get())
            if w <= 0 or h <= 0:
                raise ValueError("해상도는 1 이상이어야 합니다.")
            resolution = f"{w}x{h}"
            resolution_window.destroy()
        except ValueError as e:
            messagebox.showerror("오류", f"유효한 해상도를 입력하세요: {e}")

    submit_button = tk.Button(resolution_window, text="입력 완료", command=on_submit)
    submit_button.pack(pady=20)

    resolution_window.mainloop()
    w, h = map(int, resolution.split('x'))
    return w, h

def get_total_images():
    try:
        total_images = simpledialog.askinteger("이미지 수량 입력", "다운로드할 이미지의 수량을 입력하세요 (최대 100개):", minvalue=1, maxvalue=100)
        if not total_images:
            messagebox.showinfo("알림", "다운로드할 이미지 수량을 입력하지 않아 작업이 종료되었습니다.")
            exit()
        return total_images
    except ValueError:
        messagebox.showerror("오류", "유효한 수량을 입력해야 합니다.")
        exit()

desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
main_folder_path = os.path.join(desktop_path, "다운받은 이미지")

if not os.path.exists(main_folder_path):
    os.makedirs(main_folder_path)

@retry(requests.exceptions.RequestException, tries=1, delay=2, backoff=1.5)
def download_image(image_url, folder_path, product_name, index):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    try:
        image_response = requests.get(image_url, headers=headers, stream=True, timeout=10)
        image_response.raise_for_status()

        content_type = image_response.headers.get('Content-Type', '').lower()
        # 확장자를 jpg로 제한
        if 'jpeg' in content_type or 'jpg' in content_type:
            ext = '.jpg'
        else:
            logging.warning(f"지원하지 않는 이미지 형식: {content_type} - {image_url}")
            return None  # 지원하지 않는 형식일 경우 None 반환

        filename = f"{product_name}_{index + 1}{ext}"
        image_path = os.path.join(folder_path, get_unique_filename(folder_path, filename))
        with open(image_path, "wb") as f:
            for chunk in image_response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        logging.debug(f"이미지 다운로드 완료: {image_path}")
        return image_path
    except requests.exceptions.RequestException as e:
        logging.error(f"이미지 다운로드 실패: {image_url} - 오류: {e}")
        raise

def process_image(image_path, width, height):
    """
    1) 원본 로드 (메모리)
    2) 얼굴 제거 (옵션)
    3) 배경제거(rembg) 후 흰색으로 Flatten
    4) 비율 유지 → (너무 크면 축소, 작으면 그대로)
    5) 흰색 배경 (width x height)에 중앙 배치
    6) (옵션) 샤픈 필터 적용
    7) 최종 1회만 저장 (PNG 또는 고품질 JPEG)
    """
    try:
        with Image.open(image_path) as im:
            # RGB로 통일 (색상 모드 혼동 방지)
            im = im.convert("RGB")

            # 1) 얼굴 제거 (흰색)
            im = remove_faces_dnn(im)

            # 2) 배경제거
            im = remove_background(im)

            # 3) 비율 유지 축소(큰 경우만)
            ow, oh = im.size
            tw, th = width, height

            if ow > tw or oh > th:
                scale = min(tw / ow, th / oh)
                new_w = int(ow * scale)
                new_h = int(oh * scale)
                im = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
            else:
                new_w, new_h = ow, oh

            # 4) 흰색 캔버스에 중앙 배치
            bg = Image.new("RGB", (tw, th), (255, 255, 255))
            offset_x = (tw - new_w) // 2
            offset_y = (th - new_h) // 2
            bg.paste(im, (offset_x, offset_y))

            # 5) (옵션) 샤픈 필터 적용 (과도하면 인위적이므로 적당히)
            bg = bg.filter(ImageFilter.SHARPEN)

            # 6) 최종 저장 → PNG(무손실) 권장, or 고품질 JPEG
            #   PNG 예시:
            # bg.save(image_path, format="PNG", optimize=True)
            #   고품질 JPEG 예시:
            bg.save(image_path, format="JPEG", quality=100, optimize=False)

            logging.debug(f"최종 저장 완료: {image_path}")

    except Exception as e:
        logging.error(f"이미지 처리 실패: {image_path} - 오류: {e}")

def download_and_process_images(product_name, images, folder_path, width, height):
    # 🔹 폴더 경로에서는 '/'를 경로 구분자로 인식하지 않도록 막음
    safe_product_name = product_name.replace("/", "／")

    for index, image_info in enumerate(images):
        image_url = image_info.get("link")
        if not image_url:
            continue

        image_path = download_image(image_url, folder_path, safe_product_name, index)
        if not image_path:
            continue

        process_image(image_path, width, height)


progress_window = None
progress_var = None
progress_label = None
status_label = None

def update_progress(progress, status="진행 중"):
    if progress_window and progress_window.winfo_exists():
        progress_window.after(0, lambda: [
            progress_var.set(progress),
            progress_label.config(text=f"진행률: {progress}%"),
            status_label.config(text=f"상태: {status}")
        ])

def start_download(product_names, total_images, resolution):
    global progress_window, progress_var, progress_label, status_label
    width, height = resolution
    total_tasks = len(product_names) * total_images
    failed_images = []

    progress_window = tk.Tk()
    progress_window.title("진행 상황")
    progress_window.geometry("400x200")
    progress_window.eval('tk::PlaceWindow . center')

    progress_var = tk.IntVar()
    progress_bar = ttk.Progressbar(progress_window, maximum=100, variable=progress_var, length=300)
    progress_bar.pack(pady=20, padx=20, fill=tk.X)

    progress_label = tk.Label(progress_window, text="진행률: 0%")
    progress_label.pack()

    status_label = tk.Label(progress_window, text="상태: 작업 시작 전")
    status_label.pack(pady=5)

    update_progress(0, "작업 시작")

    try:
        cpu_count = multiprocessing.cpu_count()
        logging.debug(f"Detected CPU cores: {cpu_count}")
    except Exception as e:
        logging.error(f"Failed to detect CPU cores: {e}")
        cpu_count = 4

    with ThreadPoolExecutor(max_workers=min(8, cpu_count)) as executor:
        future_to_task = {}
        for product_name in product_names:
            images = []
            for start in range(1, total_images + 1, 10):
                params = {
                    "q": product_name,
                    "cx": cse_id,
                    "searchType": "image",
                    "num": min(10, total_images - len(images)),
                    "start": start,
                    "key": api_key
                }
                try:
                    response = requests.get(search_url, params=params, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                    response.raise_for_status()
                    results = response.json().get("items", [])
                    if not results:
                        break
                    images.extend(results)
                    if len(images) >= total_images:
                        break
                except requests.exceptions.RequestException as e:
                    logging.error(f"{product_name} 이미지 검색 실패: {e}")
                    failed_images.append(f"{product_name}: 검색 오류")
                    break

            folder_path = os.path.join(main_folder_path, product_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            for index, image_info in enumerate(images[:total_images]):
                image_url = image_info.get("link")
                if not image_url:
                    continue
                future = executor.submit(download_and_process_images, product_name, [image_info], folder_path, width, height)
                future_to_task[future] = (product_name, index)

        completed_tasks = 0
        for future in as_completed(future_to_task):
            completed_tasks += 1
            progress = int((completed_tasks / total_tasks) * 100)
            product_name, index = future_to_task[future]
            try:
                future.result()
                update_progress(progress, f"{product_name} 이미지 {index + 1} 처리 중")
            except Exception as e:
                error = f"{product_name}_{index + 1}: {str(e)}"
                logging.error(error)
                failed_images.append(error)

        if failed_images:
            error_message = "\n".join(failed_images)
            update_progress(100, "작업 완료 (오류 발생)")
            progress_window.after(0, lambda: messagebox.showerror("오류 요약", f"다음 이미지 처리에 실패했습니다:\n{error_message}"))
        else:
            update_progress(100, "작업 완료")
            progress_window.after(0, lambda: messagebox.showinfo("완료", "모든 이미지 다운로드 및 처리 작업이 완료되었습니다."))

    progress_window.after(0, progress_window.destroy)
    progress_window.mainloop()

def create_download_window(product_name):
    download_window = tk.Tk()
    download_window.title("이미지 다운로드")
    download_window.geometry("400x200")
    download_window.eval('tk::PlaceWindow . center')

    # 진행률 표시줄
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(download_window, variable=progress_var, maximum=100)
    progress_bar.pack(pady=20, padx=20, fill=tk.X)

    # 제품명 및 상태 표시
    status_label = tk.Label(download_window, text=f"{product_name} 진행 중", font=("Arial", 12))
    status_label.pack(pady=10)

    # 로그 영역
    log_area = tk.Text(download_window, height=5, width=50)
    log_area.pack(pady=10)
    log_area.insert(tk.END, "로그 (디버깅)\n")

    # 예시로 진행률 업데이트
    for i in range(101):
        progress_var.set(i)
        log_area.insert(tk.END, f"진행률: {i}%\n")
        download_window.update_idletasks()
        download_window.after(50)  # 50ms 대기

    download_window.mainloop()

if __name__ == "__main__":
    product_names = get_product_names()
    w, h = get_resolution()  # ex) 600 x 600
    total_images = get_total_images()
    start_download(product_names, total_images, (w, h))
