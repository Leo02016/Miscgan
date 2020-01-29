
d_mean_facebook = [43.45936571, 43.45961348, 86.91873142,46.0287413, 12.62521675, 43.72482552];
d_mean_wiki = [46.19831933, 46.1987395,92.39663866,44.4789916,19.47165057,47.31784946];
d_mean_p2p = [7.352091954, 7.352183908, 14.70418391, 48.5057471, 6.47324384, 7.354542111];
d_mean_bc = [7.232712993 7.233054465 14.46542599 48.658016 5.602594742 7.327182641];
d_mean_gnu = [6.459485531,6.459646302,12.91897106,47.903537,2.999035525,6.579007696];
d_mean_email = [32.98686869, 32.98787879, 65.97373737, 37.7373737, 14.77699294,33.02469136];
d_mean_CA = [5.528143484, 5.528334287, 11.05628697, 47.1495898, 5.316291492,6.455988456];

LCC_facebook = [4011, 4036, 4036, 4036, 674 4010];
LCC_wiki = [2324 2370 2380 2380 1376 2325];
LCC_p2p = [10875 10180 10875 10875 3998 10876];
LCC_bc = [5838 5420 5857 5857 2390 5853];
LCC_gnu = [6107, 5898, 6220, 6220, 1860 6107];
LCC_email = [971 897 990 990 476 972];
LCC_CA = [4158 3801 5241 5241 1693 4158];

power_law_exp_facebook = [1.316023849 1.312745609 3.506998114 2.933354414 1.666384973 1.288031234];
power_law_exp_wiki = [1.315038252 1.350747436 3.173677915 3.016901164 1.432213241 1.301216489];
power_law_exp_p2p = [1.668123249 1.662816026 1.789564977 2.922108643 1.59441742 1.614278543];
power_law_exp_bc = [1.969654201 1.680192917 1.800678639 2.91625401 1.896836941 1.748472462];
power_law_exp_gnu = [1.755340555 1.655922484 1.397344808 2.915394791 1.65098059 1.656556153];
power_law_exp_email = [1.357446088 1.351978908 3.025136123 3.014471583 1.355537713 1.330997355];
power_law_exp_CA = [1.851992718 1.476008805 1.424741338 2.92036659 1.600237408 1.632762129];


gini_facebook = [0.543073344 0.542926039 0.059037994 0.266974626 0.892011237 0.386680672];
gini_wiki = [0.575281574 0.655714275 0.058298403 0.234664748 0.831383636 0.525297367];
gini_p2p = [0.480414852 0.614853115 0.144911518 0.304730696 0.897504416 0.645076241];
gini_bc = [0.706547179 0.550404966 0.146774227 0.284043245 0.921611784 0.505317772];
gini_gnu = [0.532967739 0.600632383 0.15567061 0.286410844 0.873371093 0.446726974];
gini_email = [0.538769048 0.610396698 0.061972761 0.187946143 0.753330507 0.479655975];
gini_CA = [0.554398722 0.772601684 0.167874474 0.279759779 0.890524713 0.402433754];


algorithm = {'Original-Graph'; 'Misc-GAN'; 'E-R'; 'B-A'; 'GAE';'NetGAN'};
dataset = {'facebook','p2p','gnu','CA','bitcoin','wiki','email'};
d_mean = [d_mean_facebook;  d_mean_p2p; d_mean_gnu; d_mean_CA; d_mean_bc; d_mean_wiki;d_mean_email];
LCC = [LCC_facebook; LCC_p2p; LCC_gnu; LCC_CA; LCC_bc; LCC_wiki; LCC_email];
power_law_exp = [power_law_exp_facebook; power_law_exp_p2p; power_law_exp_gnu; power_law_exp_CA; power_law_exp_bc; power_law_exp_wiki; power_law_exp_email];
gini = [gini_facebook; gini_p2p; gini_gnu; gini_CA; gini_bc;  gini_wiki; gini_email; ];

bar(d_mean)
legend(algorithm,'Location','northwest','NumColumns',3)
set(gca,'xticklabel',dataset)
set(gca,'FontSize',26)


% bar(LCC)
% legend(algorithm,'Location','northeast')
% set(gca,'xticklabel',dataset)
% set(gca,'FontSize',18)



% bar(power_law_exp)
% legend(algorithm,'Location','northeast')
% set(gca,'xticklabel',dataset)
% set(gca,'FontSize',18)


% bar(gini)
% legend(algorithm,'Location','northeast')
% set(gca,'xticklabel',dataset)
% set(gca,'FontSize',18)
