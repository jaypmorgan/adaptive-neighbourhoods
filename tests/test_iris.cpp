.L ../src/main.cpp

void plot_iris()
{
  auto iris = read_iris("../tests/iris.data");

  std::vector<std::vector<float>> x{};
  for (auto & row : iris.x) {
    x.push_back({row[0], row[1]});
  }

  // which point is exactly the same as 56?
  for (int i = 0; i < x.size(); i++) {
    if (x[i][0] == x[56][0] && x[i][1] == x[56][1])
      std::cout << i << std::endl; 
  }

  auto neighbours = epsilon_expand(x, iris.y, 0.05);

  auto mg = new TCanvas("C1");
  mg->Range(4,1,8.5,6);
  auto g1 = new TGraph(); g1->SetMarkerStyle(20);
  auto g2 = new TGraph(); g2->SetMarkerStyle(20);
  auto g3 = new TGraph(); g3->SetMarkerStyle(20);
  auto g4 = new TGraph(); g4->SetMarkerStyle(8);

  g1->SetMarkerColor(1);
  g2->SetMarkerColor(2);
  g3->SetMarkerColor(3);

  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0, 0.05);

  // include jitter to the points
  for (int i = 0; i < 50; i++) {
    g1->SetPoint(i, iris.x[i][0]+distribution(generator), iris.x[i][1]+distribution(generator));
    g2->SetPoint(i, iris.x[i+50][0]+distribution(generator), iris.x[i+50][1]+distribution(generator));
    g3->SetPoint(i, iris.x[i+100][0]+distribution(generator), iris.x[i+100][1]+distribution(generator));
  }

  auto n = g1->GetN();
  for (int i = 0; i < n; i++) 

  for (int i = 0; i < neighbours.size(); i++) {
    auto ellipse = new TEllipse(iris.x[i][0], iris.x[i][1], neighbours[i]);
    ellipse->SetFillStyle(0);
    ellipse->Draw();
  }

  // add text labels to each of the points specifying their index
  for (int i = 0; i < x.size(); i++) {
    const char *s = std::to_string(i).c_str();
    auto t = new TText((double)x[i][0]+0.01, (double)x[i][1]+0.01, s);
    t->SetTextSize(0.02);
    t->Draw();
  }

  g1->Draw("P");
  g2->Draw("P");
  g3->Draw("P");
}


