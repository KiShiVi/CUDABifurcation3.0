a = readtable('C:\Users\KiShiVi\Desktop\mat.csv');
%b = csvread('C:\Users\KiShiVi\Desktop\mat.csv');

plot(a{:,1}, a{:,2}, 'k', 'LineWidth', 0.2);
hold on;
plot(a{:,1}, a{:,3}, 'r', 'LineWidth', 0.2);
hold on;
plot(a{:,1}, a{:,4}, 'g', 'LineWidth', 0.2);
hold on;
%plot(b(:,1), b(:,2), 'r', 'MarkerSize', 1);
