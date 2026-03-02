import numpy as np
import spe2py as spe
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class SpecLoader:
    """光谱文件读取"""

    def __init__(self,laser_wavelength=785):
        self.spe_tools = spe.load()
        self.spe_data = np.array(self.spe_tools.data)
        self.spe_file = self.spe_tools.file
        self.footer = self.spe_file.footer
        self.wavelength = np.array([float(x) for x in self.footer.SpeFormat.Calibrations.WavelengthMapping.Wavelength.cdata.split(',')])
        self.ramanshift = (1/laser_wavelength-1/self.wavelength)*1e7
        self.framenum = len(self.spe_data)

    def showorigingrepic(self,frame=0):
        plt.figure(figsize=(6, 6))
        plt.imshow(self.spe_data[frame][0], cmap='gray')
        plt.title('Original Image with Noise')
        plt.show()

    def selectroi(self,width=7,frame=0,mode='avg',backgroudoffset=20):
        roi = np.array([])
        rows_sum = np.sum(self.spe_data[frame][0],axis=1)
        max_index = np.argmax(rows_sum)
        bg_index = max_index+backgroudoffset
        if mode == 'sum':
            roi = sum(self.spe_data[frame][0][max_index-(width-1)//2:max_index+(width-1)//2])
            bg_roi = sum(self.spe_data[frame][0][bg_index-(width-1)//2:bg_index+(width-1)//2])
            roi = roi - bg_roi
        elif mode == 'avg':
            roi = sum(self.spe_data[frame][0][max_index - (width - 1) // 2:max_index + (width - 1) // 2])/width
            bg_roi = sum(self.spe_data[frame][0][bg_index - (width - 1) // 2:bg_index + (width - 1) // 2])/width
            roi = roi - bg_roi

        return roi,max_index

class RamanDataLoader:
    @staticmethod
    def check_quality(raman_shift,intensity):

        """检查数据质量"""
        stats = {
            'mean': np.mean(intensity),
            'std': np.std(intensity),
            'snr': np.mean(intensity) / np.std(intensity) if np.std(intensity) > 0 else 0,
            'range': np.ptp(raman_shift),
            'n_points': len(raman_shift)
        }
        return stats

class RamanDenoiser:
    @staticmethod
    def savgol_filter(intensity, window_length=11, polyorder=3):
        """Savitzky-Golay平滑滤波"""
        from scipy.signal import savgol_filter
        return savgol_filter(intensity, window_length, polyorder)

    @staticmethod
    def median_filter(intensity, kernel_size=3):
        """中值滤波"""
        from scipy.ndimage import median_filter
        return median_filter(intensity, size=kernel_size)

    @staticmethod
    def gaussian_filter(intensity, sigma=1):
        """高斯滤波"""
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(intensity, sigma)

class BaselineCorrector:
    @staticmethod
    def polynomial_fit(intensity, raman_shift, degree=20):
        """多项式拟合基线扣除"""
        # 选择基线点（通常选择谷底）
        from scipy.signal import argrelextrema
        minima = argrelextrema(intensity, np.less)[0]

        # 如果找不到足够的谷底点，使用均匀采样
        if len(minima) < degree + 2:
            indices = np.linspace(0, len(intensity) - 1, 10, dtype=int)
        else:
            indices = minima

        # 多项式拟合
        coefs = np.polyfit(raman_shift[indices], intensity[indices], degree)
        baseline = np.polyval(coefs, raman_shift)

        return intensity - baseline

    @staticmethod
    def airPLS(intensity, lambda_param=100, porder=1, itermax=15):
        """自适应迭代重加权惩罚最小二乘法"""
        m = len(intensity)
        w = np.ones(m)

        for i in range(itermax):
            z = BaselineCorrector._whittaker_smoother(intensity, w, lambda_param, porder)
            d = intensity - z
            dssn = np.abs(d[d < 0].sum())

            if i == 0:
                tolerance = dssn * 0.001

            w_new = w.copy()
            w_new[d > 0] = 0
            w_new[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
            w_new[d < 0] = 0 if dssn < tolerance else w_new[d < 0]

            if np.allclose(w, w_new):
                break

            w = w_new

        return intensity - z

    @staticmethod
    def _whittaker_smoother(x, w, lamb, porder):
        """Whittaker平滑器"""
        from scipy import sparse
        m = len(x)
        diff = np.diff(np.eye(m), porder)
        D = sparse.csc_matrix(diff)
        W = sparse.diags(w, 0, shape=(m, m))
        A = W + lamb * D.T * D
        from scipy.sparse.linalg import spsolve
        z = spsolve(A, w * x)
        return z

    @staticmethod
    def rubberband_baseline(intensity, raman_shift):
        """橡皮筋基线扣除"""
        from scipy.spatial import ConvexHull

        # 创建点集
        points = np.column_stack([raman_shift, intensity])

        # 计算凸包
        hull = ConvexHull(points)

        # 找到下凸包（基线）
        vertices = hull.vertices
        lower_hull_indices = []

        for simplex in hull.simplices:
            if points[simplex[0], 1] < np.median(intensity) and \
                    points[simplex[1], 1] < np.median(intensity):
                lower_hull_indices.extend(simplex)

        lower_hull_indices = np.unique(lower_hull_indices)

        # 插值得到基线
        baseline = np.interp(raman_shift,
                             raman_shift[lower_hull_indices],
                             intensity[lower_hull_indices])

        return intensity - baseline

class Normalizer:
    @staticmethod
    def minmax_normalize(intensity):
        """最小-最大归一化"""
        return (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity) + 1e-8)

    @staticmethod
    def vector_normalize(intensity):
        """向量归一化"""
        return intensity / np.linalg.norm(intensity)

    @staticmethod
    def area_normalize(intensity):
        """面积归一化"""
        area = np.trapz(np.abs(intensity))
        return intensity / area if area > 0 else intensity

    @staticmethod
    def standard_normalize(intensity):
        """标准化（z-score）"""
        return (intensity - np.mean(intensity)) / (np.std(intensity) + 1e-8)

class PeakDetector:
    @staticmethod
    def find_peaks_scipy(intensity, raman_shift=None, height=None, distance=None,
                         prominence=None, width=None):
        """使用scipy的find_peaks函数"""
        from scipy.signal import find_peaks

        peaks, properties = find_peaks(
            intensity,
            height=height,
            distance=distance,
            prominence=prominence,
            width=width
        )

        peak_info = {
            'indices': peaks,
            'positions': raman_shift[peaks] if raman_shift is not None else peaks,
            'intensities': intensity[peaks],
            'properties': properties
        }

        # 计算半高宽
        if 'widths' in properties and raman_shift is not None:
            peak_info['fwhm'] = properties['widths'] * (raman_shift[1] - raman_shift[0])
        else:
            peak_info['fwhm'] = None

        return peak_info

    @staticmethod
    def wavelet_peak_detection(intensity, wavelet='mexh', scales=np.arange(1, 32)):
        """小波变换峰检测"""
        from scipy import signal
        cwtmatr = signal.cwt(intensity, signal.ricker, scales)

        # 寻找局部极大值
        peaks = []
        for scale_idx in range(len(scales)):
            row = cwtmatr[scale_idx]
            local_maxima = signal.argrelextrema(row, np.greater)[0]
            peaks.extend(local_maxima)

        peaks = np.unique(peaks)
        return peaks

    @staticmethod
    def derivative_peak_detection(intensity, raman_shift, threshold=0):
        """导数法峰检测"""
        # 一阶导数
        dy = np.gradient(intensity, raman_shift)

        # 二阶导数
        d2y = np.gradient(dy, raman_shift)

        # 寻找峰位置（一阶导数为零且二阶导数为负）
        peaks = []
        for i in range(1, len(intensity) - 1):
            if dy[i - 1] > 0 and dy[i + 1] < 0 and d2y[i] < threshold:
                peaks.append(i)

        return np.array(peaks)

class PeakFitter:
    @staticmethod
    def gaussian_fit(raman_shift, intensity, peak_positions):
        """高斯峰拟合"""
        from scipy.optimize import curve_fit

        def gaussian(x, amp, mu, sigma):
            return amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

        def multi_gaussian(x, *params):
            y = np.zeros_like(x)
            for i in range(0, len(params), 3):
                amp, mu, sigma = params[i:i + 3]
                y += gaussian(x, amp, mu, sigma)
            return y

        # 初始参数
        initial_params = []
        bounds_lower = []
        bounds_upper = []

        for pos in peak_positions:
            idx = np.argmin(np.abs(raman_shift - pos))
            amp = intensity[idx]
            sigma = 5  # 初始猜测
            initial_params.extend([amp, pos, sigma])

            # 设置边界
            bounds_lower.extend([0, pos - 10, 1])
            bounds_upper.extend([amp * 2, pos + 10, 20])

        # 拟合
        try:
            popt, pcov = curve_fit(
                multi_gaussian, raman_shift, intensity,
                p0=initial_params,
                bounds=(bounds_lower, bounds_upper),
                maxfev=5000
            )

            # 提取每个峰的参数
            peak_params = []
            for i in range(0, len(popt), 3):
                peak_params.append({
                    'amplitude': popt[i],
                    'position': popt[i + 1],
                    'fwhm': 2.355 * popt[i + 2],  # 高斯峰FWHM = 2.355σ
                    'area': popt[i] * popt[i + 2] * np.sqrt(2 * np.pi)
                })

            # 计算拟合曲线
            fitted_curve = multi_gaussian(raman_shift, *popt)

            return {
                'parameters': peak_params,
                'fitted_curve': fitted_curve,
                'covariance': pcov
            }
        except:
            return None

    @staticmethod
    def lorentzian_fit(raman_shift, intensity, peak_positions):
        """洛伦兹峰拟合"""
        from scipy.optimize import curve_fit

        def lorentzian(x, amp, x0, gamma):
            return amp * gamma ** 2 / ((x - x0) ** 2 + gamma ** 2)

        def multi_lorentzian(x, *params):
            y = np.zeros_like(x)
            for i in range(0, len(params), 3):
                amp, x0, gamma = params[i:i + 3]
                y += lorentzian(x, amp, x0, gamma)
            return y

        # 类似高斯拟合的实现...

class FeatureExtractor:
    @staticmethod
    def extract_peak_features(intensity, raman_shift, peak_indices):
        """提取峰特征"""
        features = {}

        for i, idx in enumerate(peak_indices):
            features[f'peak_{i + 1}'] = {
                'position': raman_shift[idx],
                'intensity': intensity[idx],
                'relative_intensity': intensity[idx] / np.max(intensity),
                'area': FeatureExtractor._calculate_peak_area(intensity, raman_shift, idx),
                'fwhm': FeatureExtractor._calculate_fwhm(intensity, raman_shift, idx)
            }

        return features

    @staticmethod
    def _calculate_peak_area(intensity, raman_shift, peak_idx, window=10):
        """计算峰面积"""
        left = max(0, peak_idx - window)
        right = min(len(intensity), peak_idx + window)

        # 简单梯形法
        area = np.trapz(intensity[left:right], raman_shift[left:right])
        return area

    @staticmethod
    def _calculate_fwhm(intensity, raman_shift, peak_idx):
        """计算半高宽"""
        peak_height = intensity[peak_idx]
        half_max = peak_height / 2

        # 寻找左边界
        left_idx = peak_idx
        while left_idx > 0 and intensity[left_idx] > half_max:
            left_idx -= 1

        # 寻找右边界
        right_idx = peak_idx
        while right_idx < len(intensity) - 1 and intensity[right_idx] > half_max:
            right_idx += 1

        # 线性插值
        if left_idx > 0:
            x1, y1 = raman_shift[left_idx], intensity[left_idx]
            x2, y2 = raman_shift[left_idx + 1], intensity[left_idx + 1]
            left_bound = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1) if y2 != y1 else x1
        else:
            left_bound = raman_shift[0]

        if right_idx < len(intensity) - 1:
            x1, y1 = raman_shift[right_idx - 1], intensity[right_idx - 1]
            x2, y2 = raman_shift[right_idx], intensity[right_idx]
            right_bound = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1) if y2 != y1 else x2
        else:
            right_bound = raman_shift[-1]

        return right_bound - left_bound

    @staticmethod
    def extract_statistical_features(intensity):
        """提取统计特征"""
        return {
            'mean': np.mean(intensity),
            'std': np.std(intensity),
            'skewness': pd.Series(intensity).skew(),
            'kurtosis': pd.Series(intensity).kurtosis(),
            'energy': np.sum(intensity ** 2),
            'entropy': -np.sum(intensity * np.log(intensity + 1e-8)),
            'area_under_curve': np.trapz(intensity),
            'max_intensity': np.max(intensity),
            'min_intensity': np.min(intensity),
            'range': np.ptp(intensity)
        }

class RamanAnalyzer:
    @staticmethod
    def pca_analysis(spectra_matrix, n_components=2):
        """主成分分析"""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(spectra_matrix)

        # PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        return {
            'components': X_pca,
            'explained_variance': pca.explained_variance_ratio_,
            'loadings': pca.components_,
            'model': pca
        }

    @staticmethod
    def pls_analysis(X, y, n_components=2):
        """偏最小二乘回归"""
        from sklearn.cross_decomposition import PLSRegression

        pls = PLSRegression(n_components=n_components)
        pls.fit(X, y)

        return {
            'x_scores': pls.x_scores_,
            'y_scores': pls.y_scores_,
            'loadings': pls.x_loadings_,
            'coefficients': pls.coef_,
            'model': pls
        }

    @staticmethod
    def kmeans_clustering(spectra_features, n_clusters=3):
        """K-means聚类"""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(spectra_features)

        # 评估聚类效果
        silhouette = silhouette_score(spectra_features, labels)

        return {
            'labels': labels,
            'centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'silhouette_score': silhouette,
            'model': kmeans
        }

    @staticmethod
    def hierarchical_clustering(spectra_features, n_clusters=3):
        """层次聚类"""
        from sklearn.cluster import AgglomerativeClustering
        import scipy.cluster.hierarchy as sch

        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'
        )
        labels = clustering.fit_predict(spectra_features)

        # 生成树状图
        linkage_matrix = sch.linkage(spectra_features, method='ward')

        return {
            'labels': labels,
            'linkage_matrix': linkage_matrix,
            'model': clustering
        }

class RamanClassifier:
    @staticmethod
    def train_classifier(X_train, y_train, classifier_type='svm'):
        """训练分类器"""
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import classification_report

        if classifier_type == 'svm':
            from sklearn.svm import SVC
            model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
        elif classifier_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif classifier_type == 'knn':
            from sklearn.neighbors import KNeighborsClassifier
            model = KNeighborsClassifier(n_neighbors=5)
        elif classifier_type == 'mlp':
            from sklearn.neural_network import MLPClassifier
            model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)

        # 交叉验证
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)

        # 训练模型
        model.fit(X_train, y_train)

        return {
            'model': model,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }

    @staticmethod
    def predict_unknown(spectrum, model, scaler=None):
        """预测未知样本"""
        if scaler:
            spectrum = scaler.transform(spectrum.reshape(1, -1))

        prediction = model.predict(spectrum.reshape(1, -1))
        probability = model.predict_proba(spectrum.reshape(1, -1))

        return {
            'prediction': prediction[0],
            'probability': probability[0],
            'confidence': np.max(probability[0])
        }

class RamanSpectrumProcessor:
    def __init__(self, raman_shift, intensity):
        """初始化拉曼光谱处理器"""
        self.raman_shift = raman_shift
        self.original_intensity = intensity.copy()
        self.processed_intensity = intensity.copy()
        self.peaks = None
        self.features = None
        self.baseline = None

    def full_processing_pipeline(self):
        """完整处理流程"""
        print("=" * 60)
        print("开始拉曼光谱处理流程")
        print("=" * 60)

        # 1. 数据质量检查
        print("\n1. 数据质量检查...")
        stats = self.check_quality()
        print(f"   数据点数: {stats['n_points']}")
        print(f"   信号范围: {stats['range']:.2f} cm⁻¹")
        print(f"   信噪比: {stats['snr']:.2f}")

        # 2. 降噪
        print("\n2. 降噪处理...")
        self.denoise('savgol', window_length=20, polyorder=3)

        # 3. 基线扣除
        print("\n3. 基线扣除...")
        self.correct_baseline('polynomial', degree=5)

        # 4. 归一化
        print("\n4. 数据归一化...")
        self.normalize('minmax')

        # 5. 峰检测
        print("\n5. 峰检测...")
        self.detect_peaks(height=0.1, distance=10, prominence=0.05)

        # 6. 特征提取
        print("\n6. 特征提取...")
        self.extract_features()

        # 7. 峰拟合
        print("\n7. 峰拟合...")
        if self.peaks and len(self.peaks['positions']) > 0:
            self.fit_peaks('gaussian')

        print("\n" + "=" * 60)
        print("处理完成！")
        print("=" * 60)

        return self

    def denoise(self, method='savgol', **kwargs):
        """降噪"""
        denoiser = RamanDenoiser()

        if method == 'savgol':
            self.processed_intensity = denoiser.savgol_filter(
                self.processed_intensity,
                kwargs.get('window_length', 11),
                kwargs.get('polyorder', 3)
            )
        elif method == 'wavelet':
            self.processed_intensity = denoiser.wavelet_denoise(
                self.processed_intensity,
                kwargs.get('wavelet', 'db4'),
                kwargs.get('level', 3)
            )

        return self

    def correct_baseline(self, method='airPLS', **kwargs):
        """基线校正"""
        corrector = BaselineCorrector()

        if method == 'airPLS':
            self.baseline = self.processed_intensity - corrector.airPLS(
                self.processed_intensity,
                kwargs.get('lambda_param', 100)
            )
            self.processed_intensity = corrector.airPLS(
                self.processed_intensity,
                kwargs.get('lambda_param', 100)
            )
        elif method == 'polynomial':
            self.processed_intensity = corrector.polynomial_fit(
                self.processed_intensity,
                self.raman_shift,
                kwargs.get('degree', 3)
            )

        return self

    def normalize(self, method='minmax'):
        """归一化"""
        normalizer = Normalizer()

        if method == 'minmax':
            self.processed_intensity = normalizer.minmax_normalize(self.processed_intensity)
        elif method == 'vector':
            self.processed_intensity = normalizer.vector_normalize(self.processed_intensity)

        return self

    def detect_peaks(self, **kwargs):
        """峰检测"""
        detector = PeakDetector()
        self.peaks = detector.find_peaks_scipy(
            self.processed_intensity,
            self.raman_shift,
            **kwargs
        )
        return self

    def fit_peaks(self, method='gaussian'):
        """峰拟合"""
        if self.peaks is None or len(self.peaks['positions']) == 0:
            print("警告：未检测到峰，无法进行拟合")
            return self

        fitter = PeakFitter()

        if method == 'gaussian':
            fit_result = fitter.gaussian_fit(
                self.raman_shift,
                self.processed_intensity,
                self.peaks['positions']
            )

            if fit_result:
                self.fit_result = fit_result
                print(f"成功拟合 {len(fit_result['parameters'])} 个高斯峰")

                # 打印峰参数
                for i, param in enumerate(fit_result['parameters']):
                    print(f"  峰{i + 1}: 位置={param['position']:.1f} cm⁻¹, "
                          f"强度={param['amplitude']:.3f}, "
                          f"半高宽={param['fwhm']:.1f} cm⁻¹")

        return self

    def extract_features(self):
        """特征提取"""
        extractor = FeatureExtractor()

        # 峰特征
        if self.peaks:
            peak_features = extractor.extract_peak_features(
                self.processed_intensity,
                self.raman_shift,
                self.peaks['indices']
            )
        else:
            peak_features = {}

        # 统计特征
        stat_features = extractor.extract_statistical_features(self.processed_intensity)

        self.features = {
            'peak_features': peak_features,
            'statistical_features': stat_features
        }

        return self

    def check_quality(self):
        """检查数据质量"""
        loader = RamanDataLoader()
        return loader.check_quality(self.raman_shift,self.original_intensity)

    def visualize(self, figsize=(15, 10)):
        """可视化结果"""
        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # 原始光谱
        axes[0, 0].plot(self.raman_shift, self.original_intensity, 'b-', linewidth=1, alpha=0.7)
        axes[0, 0].set_xlabel('Raman Shift (cm⁻¹)')
        axes[0, 0].set_ylabel('Intensity')
        axes[0, 0].set_title('Original Spectrum')
        axes[0, 0].grid(True, alpha=0.3)

        # 处理后的光谱
        axes[0, 1].plot(self.raman_shift, self.processed_intensity, 'r-', linewidth=1.5)
        axes[0, 1].set_xlabel('Raman Shift (cm⁻¹)')
        axes[0, 1].set_ylabel('Normalized Intensity')
        axes[0, 1].set_title('Processed Spectrum')
        axes[0, 1].grid(True, alpha=0.3)

        # 叠加对比
        axes[0, 2].plot(self.raman_shift, self.original_intensity, 'b-', linewidth=1, alpha=0.5, label='Original')
        axes[0, 2].plot(self.raman_shift, self.processed_intensity, 'r-', linewidth=1.5, label='Processed')
        axes[0, 2].set_xlabel('Raman Shift (cm⁻¹)')
        axes[0, 2].set_ylabel('Intensity')
        axes[0, 2].set_title('Comparison')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 峰检测结果
        axes[1, 0].plot(self.raman_shift, self.processed_intensity, 'k-', linewidth=1)
        if self.peaks:
            axes[1, 0].plot(self.peaks['positions'], self.peaks['intensities'],
                            'ro', markersize=8, label='Detected Peaks')
            for pos, intensity in zip(self.peaks['positions'], self.peaks['intensities']):
                axes[1, 0].text(pos, intensity, f'{pos:.0f}',
                                fontsize=8, ha='center', va='bottom')
        axes[1, 0].set_xlabel('Raman Shift (cm⁻¹)')
        axes[1, 0].set_ylabel('Intensity')
        axes[1, 0].set_title('Peak Detection')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 峰拟合结果
        axes[1, 1].plot(self.raman_shift, self.processed_intensity, 'k-', linewidth=1, label='Data')
        if hasattr(self, 'fit_result'):
            axes[1, 1].plot(self.raman_shift, self.fit_result['fitted_curve'],
                            'r--', linewidth=2, label='Fitted')
        axes[1, 1].set_xlabel('Raman Shift (cm⁻¹)')
        axes[1, 1].set_ylabel('Intensity')
        axes[1, 1].set_title('Peak Fitting')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # 特征展示
        axes[1, 2].axis('off')
        if self.features:
            text_content = "Extracted Features:\n\n"

            # 统计特征
            stats = self.features['statistical_features']
            text_content += "Statistical Features:\n"
            text_content += f"Mean: {stats['mean']:.3f}\n"
            text_content += f"Std: {stats['std']:.3f}\n"
            text_content += f"Area: {stats['area_under_curve']:.3f}\n"
            text_content += f"Energy: {stats['energy']:.3f}\n\n"

            # 峰特征
            peaks_info = self.features['peak_features']
            if peaks_info:
                text_content += f"Detected Peaks: {len(peaks_info)}\n"
                for i, (key, peak) in enumerate(list(peaks_info.items())[:3]):  # 只显示前3个
                    text_content += f"Peak {i + 1}: {peak['position']:.1f} cm⁻¹\n"

            axes[1, 2].text(0.05, 0.95, text_content, transform=axes[1, 2].transAxes,
                            fontsize=9, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.show()

        return self

if __name__ == '__main__':
    spe_file = SpecLoader()

    intensity,max_index = spe_file.selectroi()
    raman_shift = spe_file.ramanshift
    processor = RamanSpectrumProcessor(raman_shift, intensity)

    # 运行完整处理流程
    processor.full_processing_pipeline()

    # 可视化结果
    processor.visualize()

    # 3. 获取特征
    if processor.features:
        print("\n提取的特征:")
        for key, value in processor.features['statistical_features'].items():
            print(f"  {key}: {value:.4f}")