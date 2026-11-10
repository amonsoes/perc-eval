import pyiqa
import csv

class FRQA:

    def __init__(self, attack_name, device, run_name=''):

        self.attack_name = attack_name
        # rule based
        self.mad = pyiqa.create_metric('mad', device=device)
        self.psnr = pyiqa.create_metric('psnr', device=device)
        self.ssim = pyiqa.create_metric('ssim', device=device)
        self.vif = pyiqa.create_metric('vif', device=device)

        # model based
        self.dists = pyiqa.create_metric('dists', device=device)
        self.lpips = pyiqa.create_metric('lpips', device=device)
        self.ahiq = pyiqa.create_metric('ahiq', device=device)
        self.topiq = pyiqa.create_metric('topiq_fr', device=device)
        self.pie_app = pyiqa.create_metric('pieapp', device=device)

        self.device = device

        self.l2_scores = []
        self.mad_scores = []
        self.psnr_scores = []
        self.ssim_scores = []
        self.vif_scores = []
        self.dists_scores = []
        self.lpips_scores = []
        self.ahiq_scores = []
        self.topiq_scores = []
        self.pie_app_scores = []

        if run_name != '':
            self.report_file = f'{run_name}/iqa_report.csv'
            # run_name something like 'data/perc_data/4.0/fgsm/....'

            with open(self.report_file, 'w') as report_file:
                report_obj = csv.writer(report_file)
                report_obj.writerow(['filename', 'l2', 'mad', 'ssim', 'psnr', 'ssim', 'vif', 'dists', 'lpips', 'ahiq', 'topiq_fr', 'pieapp'])
        
    
    def __call__(self, x, x_adv, paths):
        """
        x: clean reference images
        x_adv: adversarial/distorted images
        """

        assert x.ndim == 4
        assert x_adv.ndim == 4

        l2_score = self.get_l2(x, x_adv)

        # pyiqa expects distorted/target first, reference second.
        mad_score = self.mad(x_adv, x)
        psnr_score = self.psnr(x_adv, x)
        ssim_score = self.ssim(x_adv, x)
        ms_ssim_score = self.ms_ssim(x_adv, x)
        vif_score = self.vif(x_adv, x)

        dists_score = self.dists(x_adv, x)
        lpips_score = self.lpips(x_adv, x)
        ahiq_score = self.ahiq(x_adv, x)
        topiq_score = self.topiq(x_adv, x)

        self.write_to_report(
            paths,
            l2_score,
            mad_score,
            psnr_score,
            ssim_score,
            ms_ssim_score,
            vif_score,
            dists_score,
            lpips_score,
            ahiq_score,
            topiq_score,
        )


    def write_to_report(self, 
                        paths,
                        l2_scores,
                        mad_scores, 
                        psnr_scores, 
                        ssim_scores, 
                        vif_scores, 
                        dists_scores, 
                        lpips_scores, 
                        ahiq_scores, 
                        topiq_scores, 
                        pie_app_scores):
        with open(self.report_file, 'a') as report_file:
            report_obj = csv.writer(report_file)
            for f_p, l2_s, m_s, p_s, s_s, v_s, d_d, l_s, a_s, t_s, p_a_s in zip(paths,
                                                                        l2_scores,
                                                                        mad_scores, 
                                                                        psnr_scores, 
                                                                        ssim_scores, 
                                                                        vif_scores, 
                                                                        dists_scores, 
                                                                        lpips_scores, 
                                                                        ahiq_scores, 
                                                                        topiq_scores, 
                                                                        pie_app_scores):
                report_obj.writerow([f_p, l2_s.item(), m_s.item(), p_s.item(), s_s.item(), v_s.item(), d_d.item(), l_s.item(), a_s.item(), t_s.item(), p_a_s.item()])
                
    def get_l2(self, orig_img, adv_img):
        if orig_img.max() > 1:
            raise ValueError('original image is not 0 < x < 1')
        if adv_img.max() > 1:
            raise ValueError('adv image is not 0 < x < 1')
        distance = (orig_img - adv_img).pow(2).sum(dim=(1,2,3)).sqrt()
        return distance / orig_img.max()