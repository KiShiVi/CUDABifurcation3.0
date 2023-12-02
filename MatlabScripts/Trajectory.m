a = csvread('C:\Users\KiShiVi\Desktop\mat.csv');

figure(1);
plot(a(:,1), a(:,2), 'r.', 'MarkerSize', 2);

%exportgraphics(gcf,'bifuraction.png','Resolution',300);

%exportgraphics(gcf,'bifuraction.pdf','Resolution',300);

b = csvread('C:\Users\KiShiVi\Desktop\mat1.csv');

figure(2);
plot(b(:,1), b(:,2), 'r', 'MarkerSize', 1);

figure(3);
plot(a(:,2), b(:,2), 'r', 'MarkerSize', 1);

xlabel('t','interpreter','latex','FontSize', 24);
ylabel('X','interpreter','latex','FontSize', 24);
set(gcf, 'Position', [1 100 1280 640]);
set(gca,'FontSize',20);
set(gca,'TickLabelInterpreter','latex');
