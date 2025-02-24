import pyiqa
import csv

class FRQA:

    def __init__(self, attack_name, run_name=''):

        self.attack_name = attack_name
        # rule based
        self.mad = pyiqa.create_metric('mad')
        self.psnr = pyiqa.create_metric('psnr')
        self.ssim = pyiqa.create_metric('ssim')
        self.vif = pyiqa.create_metric('vif')

        # model based
        self.dists = pyiqa.create_metric('dists')
        self.lpips = pyiqa.create_metric('lpips')
        self.ahiq = pyiqa.create_metric('ahiq')
        self.topiq = pyiqa.create_metric('topiq_fr')
        self.pie_app = pyiqa.create_metric('pieapp')

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

            with open(self.report_file, 'a') as report_file:
                report_obj = csv.writer(report_file)
                report_obj.writerow(['filename', 'l2', 'mad', 'ssim', 'psnr', 'ssim', 'vif', 'dists', 'lpips', 'ahiq', 'topiq_fr', 'pieapp'])
        
    
    def __call__(self, x, x_adv):
        """
        x -> original images
        x_adv -> adversarial images
        """

        assert len(x.shape) == 4
        assert len(x_adv.shape) == 4

        mad_score = self.psnr(x, x_adv)
        psnr_score = self.mad(x, x_adv)
        ssim_score = self.ssim(x, x_adv)
        vif_score = self.vif(x, x_adv)
        dists_score = self.dists(x, x_adv)
        lpips_score = self.lpips(x, x_adv)
        ahiq_score = self.ahiq(x, x_adv)
        topiq_score = self.topiq(x, x_adv)
        pie_app_score = self.pie_app(x, x_adv)

        self.mad_scores.extend(mad_score)
        self.psnr_scores.extend(psnr_score)
        self.ssim_scores.extend(ssim_score)
        self.vif_scores.extend(vif_score)
        self.dists_scores.extend(dists_score)
        self.lpips_scores.extend(lpips_score)
        self.ahiq_scores.extend(ahiq_score)
        self.topiq_scores.extend(topiq_score)
        self.pie_app_scores.extend(pie_app_score)

        self.write_to_report(mad_score, psnr_score, ssim_score, vif_score,dists_score,lpips_score, ahiq_score, topiq_score, pie_app_score)
        return [mad_score, psnr_score, ssim_score, vif_score,dists_score,lpips_score, ahiq_score, topiq_score, pie_app_score]


    def write_to_report(self, 
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
            for f_p, m_s, p_s, s_s, v_s, d_d, l_s, a_s, t_s, p_a_s in zip(mad_scores, 
                                                                        psnr_scores, 
                                                                        ssim_scores, 
                                                                        vif_scores, 
                                                                        dists_scores, 
                                                                        lpips_scores, 
                                                                        ahiq_scores, 
                                                                        topiq_scores, 
                                                                        pie_app_scores):
                report_obj.writerow([f_p, m_s, p_s, s_s, v_s, d_d, l_s, a_s, t_s, p_a_s])
                
            

        

