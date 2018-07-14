import torch
import torch.utils.data as Data
import torchvision
from torch import nn
from torch.autograd import Variable
from torchvision import transforms

from network.Transformer import Transformer
from network.Discriminator import Discriminator
from network.vgg import Vgg16
from network.image_utils import generate_img_batch, calc_acc

import network.image_utils
import network.netutils

import config as cfg
import os


class Main(object):
    def __init__(self):
        # network
        self.T = None
        self.D = None
        self.opt_T = None
        self.opt_D = None
        self.self_regularization_loss = None
        self.local_adversarial_loss = None
        self.delta = None

        # data
        self.anime_train_loader = None
        self.animeblur_train_loader = None
        self.real_loader = None

        self.anime_train_loader = None
        self.animeblur_train_loader = None


    def build_network(self):
        print('=' * 50)
        print('Building network...')
        #self.T = Transformer(4, cfg.img_channels, nb_features=64)

        self.T = Transformer()
        self.D = Discriminator(input_features=cfg.img_channels)

        if cfg.cuda_use:
            self.T.cuda()
            self.D.cuda()

        self.opt_T = torch.optim.Adam(self.T.parameters(), lr=cfg.t_lr)
        self.opt_D = torch.optim.SGD(self.D.parameters(), lr=cfg.d_lr)
        self.self_regularization_loss = nn.L1Loss(size_average=False)
        self.local_adversarial_loss = nn.CrossEntropyLoss(size_average=True)
        self.delta = cfg.delta

        self.vgg = Vgg16(requires_grad=False)
        network.netutils.init_vgg16("./models/")

        self.vgg.load_state_dict(torch.load(os.path.join("./models/", "vgg16.weight")))

        self.vgg.cuda()

    def load_data(self):
        print('=' * 50)
        print('Loading data...')

        transform = transforms.Compose([
#            transforms.Grayscale,
            transforms.Scale((cfg.img_width, cfg.img_height)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        anime_train_folder = torchvision.datasets.ImageFolder(root=cfg.anime_path, transform=transform)
        animeblur_train_folder = torchvision.datasets.ImageFolder(root=cfg.animeblur_path, transform=transform)



        real_train_folder = torchvision.datasets.ImageFolder(root=cfg.real_path, transform=transform)



        self.anime_train_loader = Data.DataLoader(anime_train_folder, batch_size=cfg.batch_size, shuffle=True,
                                                pin_memory=True)
        self.animeblur_train_loader = Data.DataLoader(animeblur_train_folder, batch_size=cfg.batch_size, shuffle=True,
                                                pin_memory=True)

        self.real_train_loader = Data.DataLoader(real_train_folder, batch_size=cfg.batch_size, shuffle=True,
                                                pin_memory=True)

        print('anime_train_batch %d' % len(self.anime_train_loader))
        print('animeblur_train_batch %d' % len(self.animeblur_train_loader))
        real_folder = torchvision.datasets.ImageFolder(root=cfg.real_path, transform=transform)
        # real_folder.imgs = real_folder.imgs[:2000]
        self.real_loader = Data.DataLoader(real_folder, batch_size=cfg.batch_size, shuffle=True,
                                           pin_memory=True)
        print('real_batch %d' % len(self.real_loader))


    def pre_train_t(self):
        print('=' * 50)

        #device = torch.device("cuda" if args.cuda else "cpu")

        if cfg.ref_pre_path:
            print('Loading t_pre from %s' % cfg.ref_pre_path)
            self.T.load_state_dict(torch.load(cfg.ref_pre_path))
            return

        print('pre-training the refiner network %d times...' % cfg.t_pretrain)

        mse_loss = torch.nn.MSELoss()

        for index in range(cfg.t_pretrain):

            #print("aaaaaaaa")
            anime_image_batch, _ = iter(self.anime_train_loader).next()
            anime_image_batch = Variable(anime_image_batch).cuda()


            animeblur_image_batch, _ = iter(self.animeblur_train_loader).next()

            animeblur_image_batch = Variable(animeblur_image_batch).cuda()


            real_image_batch, _ = iter(self.real_train_loader).next()
            real_image_batch = Variable(real_image_batch).cuda()

            #print(real_image_batch.size())

            self.T.train()
            transreal_image_batch = self.T(real_image_batch)


            #################################
            real_features = self.vgg(real_image_batch).relu4_3
            transreal_features = self.vgg(transreal_image_batch).relu4_3
            #t_loss = self.self_regularization_loss( real_image_batch, transreal_image_batch )
            #t_loss = mse_loss(real_features, transreal_features)
            loss = torch.abs(transreal_features - real_features)

            #t_loss = loss.sum() / (cfg.batch_size * loss.mean())
            t_loss = loss.mean()
            print(t_loss.size())
            print(t_loss)
            #################################
            # t_loss = torch.div(t_loss, cfg.batch_size)
            t_loss = torch.mul(t_loss, self.delta)

            self.opt_T.zero_grad()
            t_loss.backward()
            self.opt_T.step()

            # log every `log_interval` steps
            if (index % cfg.t_pre_per == 0) or (index == cfg.t_pretrain - 1):
                # figure_name = 'refined_image_batch_pre_train_step_{}.png'.format(index)
                print('[%d/%d] (R)reg_loss: %.4f' % (index, cfg.t_pretrain, t_loss.data[0]))

                anime_image_batch, _ = iter(self.anime_train_loader).next()
                anime_image_batch = Variable(anime_image_batch, volatile=True).cuda()

                animeblur_image_batch, _ = iter(self.animeblur_train_loader).next()
                animeblur_image_batch = Variable(animeblur_image_batch, volatile=True).cuda()


                real_image_batch, _ = iter(self.real_loader).next()
                real_image_batch = Variable(real_image_batch, volatile=True).cuda()

                self.T.eval()
                ref_image_batch = self.T(real_image_batch)

                figure_path = os.path.join(cfg.train_res_path, 'refined_image_batch_pre_train_%d.png' % index)

                generate_img_batch(anime_image_batch.data.cpu(), ref_image_batch.data.cpu(),
                                   real_image_batch.data.cpu(), figure_path)



                self.T.train()

                print('Save t_pre to models/t_pre.pkl')
                torch.save(self.T.state_dict(), 'models/t_pre.pkl')

    def pre_train_d(self):
        print('=' * 50)
        if cfg.disc_pre_path:
            print('Loading D_pre from %s' % cfg.disc_pre_path)
            self.D.load_state_dict(torch.load(cfg.disc_pre_path))
            return


        print('pre-training the discriminator network %d times...' % cfg.t_pretrain)

        self.D.train()
        self.T.eval()
        for index in range(cfg.d_pretrain):
            real_image_batch, _ = iter(self.real_loader).next()
            real_image_batch = Variable(real_image_batch).cuda()

            anime_image_batch, _ = iter(self.anime_train_loader).next()
            anime_image_batch = Variable(anime_image_batch).cuda()

            animeblur_image_batch, _ = iter(self.anime_train_loader).next()
            animeblur_image_batch = Variable(animeblur_image_batch).cuda()


            assert real_image_batch.size(0) == anime_image_batch.size(0)
            assert real_image_batch.size(0) == anime_image_batch.size(0)

            d_real_pred = self.D(real_image_batch).view(-1, 2)


            d_anime_y = Variable(torch.zeros(d_real_pred.size(0)).type(torch.LongTensor)).cuda()
            # real to fake
            d_real_y = Variable(torch.ones(d_real_pred.size(0)).type(torch.LongTensor)).cuda()
            # blur is fake
            d_blur_y = Variable(torch.ones(d_real_pred.size(0)).type(torch.LongTensor)).cuda()


            # ============ real image D ====================================================
            # self.D.train()
            d_anime_pred = self.D(anime_image_batch).view(-1, 2)
            acc_anime = calc_acc(d_anime_pred, 'real')
            d_loss_anime = self.local_adversarial_loss(d_anime_pred, d_anime_y)
            # d_loss_real = torch.div(d_loss_real, cfg.batch_size)

            # ============ anime image D ====================================================
            # self.T.eval()
            real_image_batch = self.T(real_image_batch)

            # self.D.train()
            d_real_pred = self.D(real_image_batch).view(-1, 2)
            acc_real = calc_acc(d_real_pred, 'refine')
            d_loss_real = self.local_adversarial_loss(d_real_pred, d_real_y)
            # d_loss_ref = torch.div(d_loss_ref, cfg.batch_size)

            # =========== blue image D =============

            d_animeblur_pred = self.D(animeblur_image_batch).view(-1, 2)
            acc_blur = calc_acc(d_animeblur_pred, 'refine')
            d_loss_animeblur = self.local_adversarial_loss(d_animeblur_pred, d_blur_y)



            d_loss = d_loss_anime + d_loss_animeblur + d_loss_real

            self.opt_D.zero_grad()
            d_loss.backward()
            self.opt_D.step()

            if (index % cfg.d_pre_per == 0) or (index == cfg.d_pretrain - 1):
                print('[%d/%d] (D)d_loss:%f  acc_anime:%.2f%% acc_real:%.2f%% acc_blur:%.2f%%'
                      % (index, cfg.d_pretrain, d_loss.data[0], acc_anime, acc_real, acc_blur))

        print('Save D_pre to models/D_pre.pkl')
        torch.save(self.D.state_dict(), 'models/D_pre.pkl')

    def train(self):
        print('=' * 50)
        print('Training...')


        #self.D.load_state_dict(torch.load("models/D_620.pkl"))
        #self.T.load_state_dict(torch.load("models/T_620.pkl"))


        '''
        image_history_buffer = ImageHistoryBuffer((0, cfg.img_channels, cfg.img_height, cfg.img_width),
                                                  cfg.buffet_size * 10, cfg.batch_size)
        '''

        
        for step in range(cfg.train_steps):
            print('Step[%d/%d]' % (step, cfg.train_steps))

            # ========= train the T =========
            self.D.eval()
            self.T.train()

            for p in self.D.parameters():
                p.requires_grad = False

            total_t_loss = 0.0
            total_t_loss_reg_scale = 0.0
            total_t_loss_adv = 0.0
            total_acc_adv = 0.0

            for index in range(cfg.k_t):

                real_image_batch, _ = iter(self.real_loader).next()
                real_image_batch = Variable(real_image_batch).cuda()

                #real_image_batch, _ = iter(self.real_loader).next()
                #real_image_batch = Variable(real_image_batch).cuda()

                d_real_pred = self.D(real_image_batch).view(-1, 2)

                d_real_y = Variable(torch.zeros(d_real_pred.size(0)).type(torch.LongTensor)).cuda()

                transreal_image_batch = self.T(real_image_batch)
                d_transreal_pred = self.D(transreal_image_batch).view(-1, 2)

                acc_adv = calc_acc(d_transreal_pred, 'real')
                t_loss_adv = self.local_adversarial_loss(d_transreal_pred, d_real_y)

		#--------================================================================

                real_features = self.vgg(real_image_batch).relu4_3
                transreal_features = self.vgg(transreal_image_batch).relu4_3
                #t_loss = self.self_regularization_loss( real_image_batch, transreal_image_batch )
                #t_loss = mse_loss(real_features, transreal_features)
                loss = torch.abs(transreal_features - real_features)

                #t_loss_reg = loss.sum() / cfg.batch_size

                t_loss_reg = loss.mean()

                #--------================================================================

                #t_loss_reg = self.self_regularization_loss(   self.vgg(real_image_batch).relu4_3 ,  self.vgg(transreal_image_batch).relu4_3  )
                t_loss_reg_scale = torch.mul(t_loss_reg, self.delta)


                t_loss = t_loss_adv + t_loss_reg_scale

                self.opt_T.zero_grad()
                self.opt_D.zero_grad()
                t_loss.backward()
                self.opt_T.step()

                total_t_loss += t_loss
                total_t_loss_reg_scale += t_loss_reg_scale
                total_t_loss_adv += t_loss_adv
                total_acc_adv += acc_adv

            mean_t_loss = total_t_loss / cfg.k_t
            mean_t_loss_reg_scale = total_t_loss_reg_scale / cfg.k_t
            mean_t_loss_adv = total_t_loss_adv / cfg.k_t
            mean_acc_adv = total_acc_adv / cfg.k_t

            print('(R)t_loss:%.4f t_loss_reg:%.4f, t_loss_adv:%f(%.2f%%)'
                  % (mean_t_loss.data[0], mean_t_loss_reg_scale.data[0], mean_t_loss_adv.data[0], mean_acc_adv))

            # ========= train the D =========
            self.T.eval()
            self.D.train()
            for p in self.D.parameters():
                p.requires_grad = True

            for index in range(cfg.k_d):
                real_image_batch, _ = iter(self.real_loader).next()
                anime_image_batch, _ = iter(self.anime_train_loader).next()
                animeblur_image_batch, _ = iter(self.animeblur_train_loader).next()
                assert real_image_batch.size(0) == anime_image_batch.size(0)

                real_image_batch = Variable(real_image_batch).cuda()
                anime_image_batch = Variable(anime_image_batch).cuda()
                animeblur_image_batch = Variable(anime_image_batch).cuda()





                d_anime_y = Variable(torch.zeros(d_real_pred.size(0)).type(torch.LongTensor)).cuda()
                d_real_y = Variable(torch.ones(d_real_pred.size(0)).type(torch.LongTensor)).cuda()
                d_blur_y = Variable(torch.ones(d_real_pred.size(0)).type(torch.LongTensor)).cuda()

                d_anime_pred = self.D(anime_image_batch).view(-1, 2)
                acc_anime = calc_acc(d_anime_pred, 'real')
                d_loss_anime = self.local_adversarial_loss(d_anime_pred, d_anime_y)

                real_image_batch = self.T(real_image_batch)
                d_real_pred = self.D(real_image_batch).view(-1, 2)
                acc_real = calc_acc(d_real_pred, 'refine')
                d_loss_real = self.local_adversarial_loss(d_real_pred, d_real_y)

                d_animeblur_pred = self.D(animeblur_image_batch).view(-1, 2)
                acc_blur = calc_acc(d_animeblur_pred, 'refine')
                d_loss_animeblur = self.local_adversarial_loss(d_animeblur_pred, d_blur_y)





                d_loss = d_loss_real + d_loss_anime + d_loss_animeblur

                self.D.zero_grad()
                d_loss.backward()
                self.opt_D.step()

                print('(D)d_loss:%.4f anime_loss:%.4f(%.2f%%) real_loss:%.4f(%.2f%%) blur_loss:%.4f(%.2f%%)'
                      % (d_loss.data[0] / 2, d_loss_anime.data[0], acc_anime, d_loss_real.data[0], acc_real, d_loss_animeblur.data[0], acc_blur))

            if step % cfg.save_per == 0:
                print('Save two model dict.')
                torch.save(self.D.state_dict(), cfg.D_path % step)
                torch.save(self.T.state_dict(), cfg.T_path % step)


                real_image_batch, _ = iter(self.real_loader).next()
                real_image_batch = Variable(real_image_batch, volatile=True).cuda()

                anime_image_batch, _ = iter(self.anime_train_loader).next()
                anime_image_batch = Variable(anime_image_batch, volatile=True).cuda()

                animeblur_image_batch, _ = iter(self.animeblur_train_loader).next()
                animeblur_image_batch = Variable(animeblur_image_batch, volatile=True).cuda()


                self.T.eval()
                realtrans_image_batch = self.T(real_image_batch)
                self.generate_batch_train_image(real_image_batch, realtrans_image_batch, animeblur_image_batch, step_index=step)

    def generate_batch_train_image(self, anime_image_batch, ref_image_batch, real_image_batch, step_index=-1):
        print('=' * 50)
        print('Generating a batch of training images...')
        self.T.eval()

        pic_path = os.path.join(cfg.train_res_path, 'step_%d.png' % step_index)
        generate_img_batch(anime_image_batch.cpu().data, ref_image_batch.cpu().data, real_image_batch.cpu().data, pic_path)
        print('=' * 50)

    # def generate_all_train_image(self):
    #     print('=' * 50)
    #     print('Generating all training images...')
    #     self.T.eval()
    #
    #     for index, (anime_image_batch, _) in enumerate(self.anime_train_loader):
    #         pic_path = os.path.join(cfg.final_res_path, 'batch_%d.png' % index)
    #
    #         anime_image_batch = Variable(anime_image_batch, volatile=True).cuda()
    #         ref_image_batch = self.R(anime_image_batch)
    #         generate_img_batch(anime_image_batch.cpu().data, ref_image_batch.cpu().data, pic_path)
    #     print('=' * 50)


if __name__ == '__main__':
    obj = Main()

    if not torch.cuda.is_available():
        raise ValueError("ERROR: cuda is not available, try running on CPU")

    obj.build_network()

    obj.load_data()

#    obj.pre_train_t()
#    obj.pre_train_d()
    obj.train()

    obj.generate_all_train_image()


