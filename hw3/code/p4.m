y = 1;
pos = [0 1 2 7];
neg = [4 5 9 10];
plot(pos, y, '^k');
hold on;
plot(neg, y, 'xk');
set(gca, 'ytick', []);
grid on;
%%
plot([3 3], [0 2], '-b', 'LineWidth', 2);
xlabel('x');
text(3.1, 1.3, '(1)');
text(3.4, 1.2, '-');
text(2.5, 1.2, '+');
%%
plot([8 8], [0 2], '-b', 'LineWidth', 2);
xlabel('x');
text(8.1, 1.3, '(2)');
text(8.4, 1.2, '-');
text(7.5, 1.2, '+');
%%
plot([8 8], [0 2], '-b', 'LineWidth', 2);
xlabel('x');
text(8.1, 1.3, '(3)');
text(8.4, 1.2, '-');
text(7.5, 1.2, '+');