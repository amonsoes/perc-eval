
import random
import numpy as np
import torch


from scipy.fftpack import dct, idct
from src.adversarial.attack_base import Attack



class PPBA(Attack):

    def __init__(self,
                model, 
                model_trms, 
                low_dim=1500, 
                freq_dim=28, 
                stride=7,
                mom=1,
                order='strided',
                r=2352,
                n_queries=2000,
                rho=0.01,
                *args, 
                **kwargs):
        super().__init__('PPBA', model, model_trms, *args, **kwargs)
        self.low_dim=low_dim
        self.freq_dim=freq_dim 
        self.stride=stride
        self.mom=mom
        self.order=order
        self.r=r
        self.max_iter=n_queries
        self.rho=rho
        self.total_queries = []
        self.total_success = []
        self.total_l2 = 0

        image_size = 299 if self.model.model_name == 'inception' else 224
        self.image_size = image_size

        if self.model.model_name == "inception":
            self.freq_dim = 38
            self.stride = 9

        if self.order == "strided":
            Random_Matrix = np.zeros((self.low_dim, 3 * image_size * image_size))
            indices = block_order(image_size, 3, initial_size=self.freq_dim, stride=self.stride)
        else:
            Random_Matrix = np.zeros((self.low_dim, 3 * self.freq_dim * self.freq_dim))
            indices = random.sample(range(self.r), self.low_dim)
        for i in range(self.low_dim):
            Random_Matrix[i][indices[i]] = 1

        if self.order == "strided":
            Random_Matrix = torch.from_numpy(Random_Matrix).view(-1, 3, image_size, image_size)
        else:
            Random_Matrix = torch.from_numpy(Random_Matrix).view(-1, 3, self.freq_dim, self.freq_dim)


        self.Random_Matrix = (
            block_idct(
                self.expand_vector(
                    Random_Matrix, size=(image_size if self.order == "strided" else self.freq_dim),
                ),
                block_size=image_size,
            )
            .view(self.low_dim, -1)
            .to(self.device)
        )

    def forward(self, x, y):
        n_samples = x.shape[0]
        N_query = 0
        z = np.zeros((1, self.low_dim))
        prev_f = self.get_loss(z, x, y, N_query)[0]
        is_success = 0 if prev_f > 0 else 1
        effective_number = [
            np.ones((1, self.low_dim)),
            np.ones((1, self.low_dim)),
            np.ones((1, self.low_dim)),
        ]
        ineffective_number = [
            np.ones((1, self.low_dim)),
            np.ones((1, self.low_dim)),
            np.ones((1, self.low_dim)),
        ]

        for k in range(self.max_iter):
            print(f'iter:{k}, current_loss:{prev_f}')
            u = np.zeros((n_samples, self.low_dim))
            r = np.random.uniform(size=(n_samples, self.low_dim))
            effective_probability = [
                effective_number[i] / (effective_number[i] + ineffective_number[i])
                for i in range(len(effective_number))
            ]
            probability = PPBA.get_probability(effective_probability)

            # u[r < probability[0] + probability[1]] = 0
            u[r < probability[0]] = -1
            u[r >= probability[0] + probability[1]] = 1

            uz = z + self.rho * u

            uz_l2 = np.linalg.norm(uz, axis=1)
            uz = uz * np.minimum(1, 5 / uz_l2).reshape(-1, 1)

            fu = self.get_loss(uz, x, y, N_query)
            if fu.min() < prev_f:
                worked_u = u[fu < prev_f] # TODO: some issue here
                effective_probability[0] = effective_probability[0] * self.mom + (
                    worked_u == -1
                ).sum(0)
                effective_probability[1] = effective_probability[1] * self.mom + (
                    worked_u == 0
                ).sum(0)
                effective_probability[2] = effective_probability[2] * self.mom + (
                    worked_u == 1
                ).sum(0)
                not_worked_u = u[fu >= prev_f]
                ineffective_number[0] = ineffective_number[0] * self.mom + (
                    not_worked_u == -1
                ).sum(0)
                ineffective_number[1] = ineffective_number[1] * self.mom + (
                    not_worked_u == 0
                ).sum(0)
                ineffective_number[2] = ineffective_number[2] * self.mom + (
                    not_worked_u == 1
                ).sum(0)
                z = uz[np.argmin(fu)]
                prev_f = fu.max()
            else:
                ineffective_number[0] += (u == -1).sum(0)
                ineffective_number[1] += (u == 0).sum(0)
                ineffective_number[2] += (u == 1).sum(0)

            if prev_f <= 0:
                is_success = 1
                break
        
        current_q = k
        N_query = 0

        z = torch.from_numpy(z).float().to(self.device)
        perturbation = (z @ self.Random_Matrix).view(1, 3, self.image_size, self.image_size)
        new_image = (x + perturbation).clamp(0, 1)
        if current_q > self.max_iter:
            is_success = 0
        print(current_q, is_success, perturbation.view(1, -1).norm(2, 1).item())

        self.total_queries.append(current_q,)
        self.total_success.append(is_success)
        self.total_l2 += perturbation.view(1, -1).norm(2, 1).item()
        return new_image, current_q

    def cw_loss(self, x, y, N_query, targeted=False):
        N_query += x.shape[0]
        outputs = self.model(self.model_trms(x))
        one_hot_labels = torch.eye(len(outputs[0]))[y].to(self.device)

        i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
        j, _ = torch.max((one_hot_labels) * outputs, dim=1)

        if targeted:
            return torch.clamp(i - j, min=0)

        else:
            return torch.clamp(j - i, min=0)

    def get_loss(self, zs, x, y, N_query):
        z = torch.from_numpy(zs).float().to(self.device).view(-1, self.low_dim)
        perturbation = (z @ self.Random_Matrix).view(z.shape[0], 3, self.image_size, self.image_size)
        new_image = (x + perturbation).clamp(0, 1)
        loss = self.cw_loss(new_image, y, N_query, targeted=self.targeted)
        loss = loss.cpu().numpy()
        return loss

    def get_probability(success_probability):
        probability = [v / sum(success_probability) for v in success_probability]
        return probability

    def expand_vector(self, x, size):
        batch_size = x.size(0)
        x = x.view(-1, 3, size, size)
        z = torch.zeros(batch_size, 3, self.image_size, self.image_size)
        z[:, :, :size, :size] = x
        return z


########### utils #############

# defines a block order, starting with top-left (initial_size x initial_size) submatrix
# expanding by stride rows and columns whenever exhausted
# randomized within the block and across channels
# e.g. (initial_size=2, stride=1)
# [1, 3, 6]
# [2, 4, 9]
# [5, 7, 8]

def block_order(image_size, channels, initial_size=1, stride=1):
    order = torch.zeros(channels, image_size, image_size)
    total_elems = channels * initial_size * initial_size
    perm = torch.randperm(total_elems)
    order[:, :initial_size, :initial_size] = perm.view(
        channels, initial_size, initial_size
    )
    for i in range(initial_size, image_size, stride):
        num_elems = channels * (2 * stride * i + stride * stride)
        perm = torch.randperm(num_elems) + total_elems
        num_first = channels * stride * (stride + i)
        order[:, : (i + stride), i : (i + stride)] = perm[:num_first].view(
            channels, -1, stride
        )
        order[:, i : (i + stride), :i] = perm[num_first:].view(channels, stride, -1)
        total_elems += num_elems
    return order.view(1, -1).squeeze().long().sort()[1]



# applies DCT to each block of size block_size
def block_dct(x, block_size=8, masked=False, ratio=0.5):
    z = torch.zeros(x.size())
    num_blocks = int(x.size(2) / block_size)
    mask = np.zeros((x.size(0), x.size(1), block_size, block_size))
    mask[:, :, : int(block_size * ratio), : int(block_size * ratio)] = 1
    for i in range(num_blocks):
        for j in range(num_blocks):
            submat = x[
                :,
                :,
                (i * block_size) : ((i + 1) * block_size),
                (j * block_size) : ((j + 1) * block_size),
            ].numpy()
            submat_dct = dct(dct(submat, axis=2, norm="ortho"), axis=3, norm="ortho")
            if masked:
                submat_dct = submat_dct * mask
            submat_dct = torch.from_numpy(submat_dct)
            z[
                :,
                :,
                (i * block_size) : ((i + 1) * block_size),
                (j * block_size) : ((j + 1) * block_size),
            ] = submat_dct
    return z


# applies IDCT to each block of size block_size
def block_idct(x, block_size=8, masked=False, ratio=0.5):
    z = torch.zeros(x.size())
    num_blocks = int(x.size(2) / block_size)
    mask = np.zeros((x.size(0), x.size(1), block_size, block_size))
    if type(ratio) != float:
        for i in range(x.size(0)):
            mask[i, :, : int(block_size * ratio[i]), : int(block_size * ratio[i])] = 1
    else:
        mask[:, :, : int(block_size * ratio), : int(block_size * ratio)] = 1
    for i in range(num_blocks):
        for j in range(num_blocks):
            submat = x[
                :,
                :,
                (i * block_size) : ((i + 1) * block_size),
                (j * block_size) : ((j + 1) * block_size),
            ].numpy()
            if masked:
                submat = submat * mask
            z[
                :,
                :,
                (i * block_size) : ((i + 1) * block_size),
                (j * block_size) : ((j + 1) * block_size),
            ] = torch.from_numpy(
                idct(idct(submat, axis=3, norm="ortho"), axis=2, norm="ortho")
            )
    return z




"""# get most likely predictions and probabilities for a set of inputs
def get_preds(
    model, inputs, dataset_name, correct_class=None, batch_size=25, return_cpu=True
):
    num_batches = int(math.ceil(inputs.size(0) / float(batch_size)))
    softmax = torch.nn.Softmax()
    all_preds, all_probs = None, None
    for i in range(num_batches):
        upper = min((i + 1) * batch_size, inputs.size(0))
        input = apply_normalization(inputs[(i * batch_size) : upper], dataset_name)
        input_var = torch.autograd.Variable(input.cuda(), volatile=True)
        output = softmax.forward(model.forward(input_var))
        if correct_class is None:
            prob, pred = output.max(1)
        else:
            prob, pred = (
                output[:, correct_class],
                torch.autograd.Variable(torch.ones(output.size()) * correct_class),
            )
        if return_cpu:
            prob = prob.data.cpu()
            pred = pred.data.cpu()
        else:
            prob = prob.data
            pred = pred.data
        if i == 0:
            all_probs = prob
            all_preds = pred
        else:
            all_probs = torch.cat((all_probs, prob), 0)
            all_preds = torch.cat((all_preds, pred), 0)
    return all_preds, all_probs"""


"""# get least likely predictions and probabilities for a set of inputs
def get_least_likely(model, inputs, dataset_name, batch_size=25, return_cpu=True):
    num_batches = int(math.ceil(inputs.size(0) / float(batch_size)))
    softmax = torch.nn.Softmax()
    all_preds, all_probs = None, None
    for i in range(num_batches):
        upper = min((i + 1) * batch_size, inputs.size(0))
        input = apply_normalization(inputs[(i * batch_size) : upper], dataset_name)
        input_var = torch.autograd.Variable(input.cuda(), volatile=True)
        output = softmax.forward(model.forward(input_var))
        prob, pred = output.min(1)
        if return_cpu:
            prob = prob.data.cpu()
            pred = pred.data.cpu()
        else:
            prob = prob.data
            pred = pred.data
        if i == 0:
            all_probs = prob
            all_preds = pred
        else:
            all_probs = torch.cat((all_probs, prob), 0)
            all_preds = torch.cat((all_preds, pred), 0)
    return all_preds, all_probs"""

"""
# defines a diagonal order
# order is fixed across diagonals but are randomized across channels and within the diagonal
# e.g.
# [1, 2, 5]
# [3, 4, 8]
# [6, 7, 9]
def diagonal_order(image_size, channels):
    x = torch.arange(0, image_size).cumsum(0)
    order = torch.zeros(image_size, image_size)
    for i in range(image_size):
        order[i, : (image_size - i)] = i + x[i:]
    for i in range(1, image_size):
        reverse = order[image_size - i - 1].index_select(
            0, torch.LongTensor([i for i in range(i - 1, -1, -1)])
        )
        order[i, (image_size - i) :] = image_size * image_size - 1 - reverse
    if channels > 1:
        order_2d = order
        order = torch.zeros(channels, image_size, image_size)
        for i in range(channels):
            order[i, :, :] = 3 * order_2d + i
    return order.view(1, -1).squeeze().long().sort()[1]
"""


"""# zeros all elements outside of the top-left (block_size * ratio) submatrix for every block
def block_zero(x, block_size=8, ratio=0.5):
    z = torch.zeros(x.size())
    num_blocks = int(x.size(2) / block_size)
    mask = torch.zeros(x.size(0), x.size(1), block_size, block_size)
    mask[:, :, : int(block_size * ratio), : int(block_size * ratio)] = 1
    for i in range(num_blocks):
        for j in range(num_blocks):
            z[
                :,
                :,
                (i * block_size) : ((i + 1) * block_size),
                (j * block_size) : ((j + 1) * block_size),
            ] = (
                x[
                    :,
                    :,
                    (i * block_size) : ((i + 1) * block_size),
                    (j * block_size) : ((j + 1) * block_size),
                ]
                * mask
            )
    return z"""



"""NUM = args.num
for i in tqdm(range(NUM)):
    index = i
    image = testset[index][0].unsqueeze(0).cuda()
    label = model(normalize(image)).argmax(1)

    with torch.no_grad():
        q, s, l2, _ = PPBA(model, image, label, i)

    print(i + 1, sum(total_queries) / (i + 1), sum(total_success) / (i + 1))
    print("=" * 10)


act_qs = [
    0 if total_success[i] == 0 or total_queries[i] > self.max_iter else 1
    for i in range(len(total_success))
]
sum_qs = sum([act_qs[i] * total_queries[i] for i in range(len(total_queries))])
print("ASR:", sum(act_qs) / NUM)
print("On success:")
print("average queries:", sum_qs / sum(act_qs))

print("On all:")
print("average queries:", sum(total_queries) / NUM)"""