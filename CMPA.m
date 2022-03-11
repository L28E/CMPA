clear
close all

%% Task 1

Is=0.01e-12;
Ib=0.1e-12;
Vb=1.3;
Gp=0.1;

V=linspace(-1.95,0.7,200);
I=Is*(exp(1.2*V/25e-3)-1) + Gp*V - Ib*(exp(1.2*(-(V+Vb))/25e-3)-1);
Irand = I .* (0.8+(1.2-0.8).*rand(1,length(I)));

figure(1);
subplot(2,2,1)
plot(V,I);
xlabel("V")
ylabel("I")
title("Diode Current")
subplot(2,2,2)
plot(V,Irand);
xlabel("V")
ylabel("I")
title("Diode Current /w Random Variation")
subplot(2,2,3)
semilogy(V,abs(I));
xlabel("V")
ylabel("I")
title("Semilog Diode Current")
subplot(2,2,4)
semilogy(V,abs(Irand));
xlabel("V")
ylabel("I")
title("Semilog Diode Current /w Random Variation")

%% Task 2
fit4=polyval(polyfit(V,I,4),V);
fit4Rand=polyval(polyfit(V,Irand,4),V);
fit8=polyval(polyfit(V,I,8),V);
fit8Rand=polyval(polyfit(V,Irand,8),V);

subplot(2,2,1);
hold on;
plot(V,fit4);
plot(V,fit8);
legend("current","4th order poly","8th order poly")

subplot(2,2,2);
hold on;
plot(V,fit4Rand);
plot(V,fit8Rand);
legend("current","4th order poly","8th order poly")

subplot(2,2,3);
hold on;
semilogy(V,abs(fit4));
semilogy(V,abs(fit8));
legend("current","4th order poly","8th order poly")

subplot(2,2,4);
hold on;
semilogy(V,abs(fit4Rand));
semilogy(V,abs(fit8Rand));
legend("current","4th order poly","8th order poly")

% As we increase the order of the polynomial, it converges on the current,
% but it's an exercise in diminishing returns. On a linear scale we can get 
% somehting which matches quite well, but differences between the 
% polynomial and the current are greatly exagerated on a logarithmic scale. 

%% Task 3

fn='%s.*(exp(1.2*x/25e-3)-1) + %s.*x - %s*(exp(1.2*(-(x+%s))/25e-3)-1)';

% a)
fo2 = fittype(sprintf(fn,"A",Gp,"C",Vb));
ff2 = fit(V',I',fo2);
If2 = ff2(V);

% b)
fo3 = fittype(sprintf(fn,"A","B","C",Vb));
ff3 = fit(V',I',fo3);
If3 = ff3(V);

% c)
fo4 = fittype(sprintf(fn,"A","B","C","D"));
ff4 = fit(V',I',fo4);
If4 = ff4(V);

figure(2)
hold on;
plot(V,I);
plot(V,Irand);
plot(V,If2);
plot(V,If3);
plot(V,If4);
xlabel("V")
ylabel("I")
legend("current","rand current","2 fitted","3 fitted","4 fitted")

% The more parameters we have to determine programmatically, the less
% accurate the fit/model is.

%% Task 4

inputs = V.';
targets = I.';
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs)
view(net)
Inn = outputs

figure(3)
hold on;
plot(V,I);
plot(V,Irand);
plot(V,Inn);
xlabel("V")
ylabel("I")
legend("current","rand current","NN fitted")

% Gives us a very accurate fit/model, but is a lot more computationally complex
