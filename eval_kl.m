
degree_facebook = [];
degree_wiki = [];
degree_p2p = [];
degree_bc = [];
degree_gnu = [];
degree_CA = [];
degree_email = [8.56E-01 4.238143566 2.911736982 1.09E+00 1.03E+00];

coef_facebook = [0.021573461 0.06607476 0.04826719 0.134933416 0.281401619];
coef_wiki = [0.020891959 0.284033178 0.174050588 0.354474684 0.436397701];
coef_p2p = [1.3749774 2.030172819 2.117149732 146.5213795 151.6558971];
coef_bc = [1.31982508 1.86848131 1.633203178 1.75984174 2.064657431];
coef_gnu = [0.006329814 0.002123987 0.002292609 0.04683803 0.045804066];
coef_CA = [0.289412817 0.244474165 0.226563393 0.219983906 0.357911598];
coef_email = [5.62E-02 2.16E-01 0.169512849 0.152744912 0.293283729];

kernel_facebook = [8.93E-08 9.76E-08 9.68E-08 4.78E-08 8.92E-08];
kernel_wiki = [3.83E-07 3.60E-07 3.61E-07 2.09E-07 3.59E-07];
kernel_p2p = [8.43E-09 1.21E-08 1.20E-08 1.50E-09 1.17E-08];
kernel_bc = [1.83E-07 4.65E-08 4.63E-08 2.96E-08 3.89E-08];
kernel_gnu = [2.91E-08 3.55E-08 3.53E-08 8.81E-09 3.38E-08];
kernel_CA = [3.03E-08 6.16E-08 6.14E-08 1.73E-08 4.79E-08];
kernel_email = [1.83E-06 2.02E-06 2.02E-06 9.97E-07 2.00E-06];

algorithm = {'Misc-GAN'; 'E-R'; 'B-A'; 'GAE';'NetGAN'};
coef = [coef_facebook; coef_p2p; coef_gnu; coef_CA; coef_bc; coef_wiki; coef_email;];
kernel = [kernel_facebook;  kernel_p2p; kernel_gnu; kernel_CA; kernel_bc;kernel_wiki; kernel_email];
dataset = {'facebook','p2p','gnu','CA','bitcoin','wiki','email'};

bar(-log(coef))
title('The KL divergence of clustering coefficient')
legend(algorithm,'Location','northeast')
set(gca,'xticklabel',dataset)
set(gca,'FontSize',18)
ylabel('-log()')

bar(-log(kernel))
title('Random walk graph kernel')
legend(algorithm,'Location','northeast')
set(gca,'xticklabel',dataset)
set(gca,'FontSize',18)
ylabel('-log()')