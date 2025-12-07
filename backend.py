import cv2
import numpy as np
import pywt
from scipy.fftpack import dct, idct

class RobustWatermark:
    def __init__(self, block_size=8, alpha=50, key=42):
        self.block_size = block_size
        self.alpha = alpha
        self.key = key
        
        # Optimized ZigZag indices for the "Middle Frequencies"
        # These are robust against JPEG compression and Noise
        self.mid_band_indices = [
            (1, 2), (2, 1), (2, 2), (3, 0), (0, 3), (3, 1), (1, 3), (3, 2),
            (2, 3), (4, 0), (0, 4), (4, 1), (1, 4), (2, 4), (3, 3), (4, 2),
            (5, 0), (0, 5), (5, 1), (1, 5), (4, 3), (3, 4)
        ]

    def apply_dct(self, image_block):
        return dct(dct(image_block.T, norm='ortho').T, norm='ortho')

    def apply_idct(self, dct_block):
        return idct(idct(dct_block.T, norm='ortho').T, norm='ortho')

    def generate_pn_sequences(self):
        """Generates two distinct random sequences for bit 0 and bit 1."""
        np.random.seed(self.key)
        pn0 = np.random.randn(len(self.mid_band_indices))
        pn1 = np.random.randn(len(self.mid_band_indices))
        return pn0, pn1

    def sift_geometry_correction(self, original_img, attacked_img):
        """Corrects Rotation/Scaling using SIFT Feature Matching."""
        try:
            # SIFT works on Grayscale
            if len(original_img.shape) > 2:
                img1_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            else:
                img1_gray = original_img

            if len(attacked_img.shape) > 2:
                img2_gray = cv2.cvtColor(attacked_img, cv2.COLOR_BGR2GRAY)
            else:
                img2_gray = attacked_img

            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(img1_gray, None)
            kp2, des2 = sift.detectAndCompute(img2_gray, None)

            if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
                return attacked_img

            # FLANN Matcher parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            matches = flann.knnMatch(des1, des2, k=2)

            # Lowe's Ratio Test (Strict filtering)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            if len(good_matches) > 4:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Compute Homography Matrix
                M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                if M is not None:
                    h, w = img1_gray.shape
                    corrected_img = cv2.warpPerspective(attacked_img, M, (w, h))
                    return corrected_img
        except Exception as e:
            print(f"SIFT Warning: {e}")
        
        return attacked_img

    def embed(self, cover_image, watermark_image):
        """Embeds watermark using Y-Channel DWT-DCT with Redundancy."""
        # 1. Convert to YCbCr (Embed in Y channel)
        ycbcr = cv2.cvtColor(cover_image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycbcr)
        y = y.astype(np.float32)
        h, w = y.shape

        # 2. Resize Watermark to 32x32 (Small, robust payload)
        target_dim = 32
        wm_resized = cv2.resize(watermark_image, (target_dim, target_dim))
        _, wm_binary = cv2.threshold(wm_resized, 128, 1, cv2.THRESH_BINARY)
        wm_flat = wm_binary.flatten()
        total_bits = len(wm_flat)

        # 3. DWT Transform (Haar)
        coeffs = pywt.dwt2(y, 'haar')
        LL, (LH, HL, HH) = coeffs
        
        # 4. Calculate Redundancy (Blocks per Bit)
        h_sub, w_sub = HL.shape
        num_blocks_h = h_sub // self.block_size
        num_blocks_w = w_sub // self.block_size
        total_blocks = num_blocks_h * num_blocks_w
        
        # Voting Factor: How many blocks will carry the same bit?
        self.blocks_per_bit = total_blocks // total_bits
        
        if self.blocks_per_bit < 1:
            raise ValueError(f"Image too small! Need {total_bits} blocks, have {total_blocks}.")

        # 5. Embedding Loop
        pn0, pn1 = self.generate_pn_sequences()
        
        block_iter = 0
        for bit_idx in range(total_bits):
            bit = wm_flat[bit_idx]
            sequence = pn1 if bit == 1 else pn0
            
            # Embed this bit into multiple blocks (Redundancy)
            for _ in range(self.blocks_per_bit):
                # Calculate block coordinates
                bi = block_iter // num_blocks_w
                bj = block_iter % num_blocks_w
                
                # Pixel coordinates in subband
                r_start = bi * self.block_size
                c_start = bj * self.block_size
                
                # Extract Block
                block = HL[r_start:r_start+self.block_size, c_start:c_start+self.block_size]
                
                # Forward DCT
                dct_block = self.apply_dct(block)
                
                # Add PN Sequence to Mid-Band
                for k, (rr, cc) in enumerate(self.mid_band_indices):
                    dct_block[rr, cc] += self.alpha * sequence[k]
                
                # Inverse DCT
                HL[r_start:r_start+self.block_size, c_start:c_start+self.block_size] = self.apply_idct(dct_block)
                
                block_iter += 1

        # 6. Inverse DWT and Merge
        coeffs_mod = (LL, (LH, HL, HH))
        y_watermarked = pywt.idwt2(coeffs_mod, 'haar')
        y_watermarked = np.clip(y_watermarked, 0, 255).astype(np.uint8)
        
        watermarked_bgr = cv2.cvtColor(cv2.merge([y_watermarked, cr, cb]), cv2.COLOR_YCrCb2BGR)
        
        # Metrics
        mse = np.mean((cover_image - watermarked_bgr) ** 2)
        psnr = 10 * np.log10(255**2 / mse) if mse > 0 else 100

        return watermarked_bgr, target_dim, wm_binary, psnr

    def extract(self, watermarked_img, original_img, wm_dim):
        """Extracts watermark with SIFT correction and Voting Mechanism."""
        # 1. SIFT Correction
        if original_img is not None:
            if original_img.shape != watermarked_img.shape:
                watermarked_img = cv2.resize(watermarked_img, (original_img.shape[1], original_img.shape[0]))
            
            watermarked_img = self.sift_geometry_correction(original_img, watermarked_img)

        # 2. Extract Y Channel
        ycbcr = cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2YCrCb)
        y, _, _ = cv2.split(ycbcr)
        y = y.astype(np.float32)

        # 3. DWT
        coeffs = pywt.dwt2(y, 'haar')
        _, (_, HL, _) = coeffs

        # 4. Calculate Redundancy Parameters
        h_sub, w_sub = HL.shape
        num_blocks_h = h_sub // self.block_size
        num_blocks_w = w_sub // self.block_size
        total_blocks = num_blocks_h * num_blocks_w
        total_bits = wm_dim * wm_dim
        
        # Recalculate redundancy
        if total_bits > 0:
            blocks_per_bit = total_blocks // total_bits
        else:
            blocks_per_bit = 1 # Fallback

        pn0, pn1 = self.generate_pn_sequences()
        extracted_bits = []
        
        block_iter = 0
        
        # 5. Extraction Loop with Voting
        for _ in range(total_bits):
            vote_0 = 0.0
            vote_1 = 0.0
            
            # Gather votes from all redundant blocks
            for _ in range(blocks_per_bit):
                if block_iter >= total_blocks: break

                bi = block_iter // num_blocks_w
                bj = block_iter % num_blocks_w
                r_start = bi * self.block_size
                c_start = bj * self.block_size
                
                block = HL[r_start:r_start+self.block_size, c_start:c_start+self.block_size]
                
                # DCT
                dct_block = self.apply_dct(block)
                
                # Extract Coefficients
                extracted_seq = [dct_block[rr, cc] for rr, cc in self.mid_band_indices]
                
                # Correlation
                corr0 = np.corrcoef(extracted_seq, pn0)[0, 1]
                corr1 = np.corrcoef(extracted_seq, pn1)[0, 1]
                
                # Accumulate Votes
                if np.isnan(corr0): corr0 = 0
                if np.isnan(corr1): corr1 = 0
                
                vote_0 += corr0
                vote_1 += corr1
                
                block_iter += 1
            
            # Final Decision
            if vote_1 > vote_0:
                extracted_bits.append(1)
            else:
                extracted_bits.append(0)

        extracted_bits = np.array(extracted_bits)
        # Handle Padding if needed
        if len(extracted_bits) < total_bits:
            extracted_bits = np.pad(extracted_bits, (0, total_bits - len(extracted_bits)), 'constant')
        elif len(extracted_bits) > total_bits:
            extracted_bits = extracted_bits[:total_bits]
            
        return extracted_bits.reshape((wm_dim, wm_dim)) * 255