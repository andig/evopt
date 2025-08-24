package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/andig/evopt/client"
	"github.com/guptarohit/asciigraph"
	"github.com/olekukonko/tablewriter"
	"github.com/olekukonko/tablewriter/tw"
	"github.com/samber/lo"
)

func main() {
	vFlag := flag.Bool("v", false, "verbose output")
	cwFlag := flag.Int("cw", 150, "chart width")
	chFlag := flag.Int("ch", 20, "chart height")
	jsonData := flag.String("json", "", "json request")
	uri := flag.String("uri", "http://localhost:7050", "optimizer uri")
	flag.Parse()

	if fi, _ := os.Stdin.Stat(); fi.Mode()&os.ModeNamedPipe != 0 || fi.Mode()&os.ModeCharDevice == 0 {
		data, err := io.ReadAll(os.Stdin)
		if err != nil {
			log.Fatal(err)
		}
		*jsonData = string(data)
	}

	// custom HTTP client
	hc := http.Client{
		Timeout: 10 * time.Second,
	}

	c, err := client.NewClientWithResponses(*uri, client.WithHTTPClient(&hc))
	if err != nil {
		log.Fatal(err)
	}

	var req client.OptimizationInput
	if *jsonData != "" {
		if err := json.Unmarshal([]byte(*jsonData), &req); err != nil {
			log.Fatal(err)
		}
	} else {
		example, err := c.GetOptimizeExampleWithResponse(context.TODO())
		if err != nil {
			log.Fatal(err)
		}

		if example.StatusCode() != http.StatusOK {
			log.Fatalf("Expected HTTP 200 but received %d\n%s", example.StatusCode(), string(example.Body))
		}

		req := *example.JSON200
		if *vFlag {
			b, _ := json.MarshalIndent(req, "", "  ")
			fmt.Println(string(b))
		}
	}

	tw := tablewriter.WithConfig(tablewriter.Config{
		Row: tw.CellConfig{
			Alignment: tw.CellAlignment{Global: tw.AlignRight},
		},
	})

	{
		table := tablewriter.NewTable(os.Stdout, tw)
		headers := []string{"Hour", "Forecast", "TotalDemand", "GridImportCost", "GridExportCost"}

		for i, bat := range req.Batteries {
			if bat.SGoal != nil && lo.Sum(*bat.SGoal) > 0 {
				headers = append(headers,
					fmt.Sprintf("Bat %d Goal", i),
				)
			}
		}

		table.Header(headers)

		for t := range len(req.TimeSeries.Ft) {
			row := []string{
				strconv.Itoa(t + 1),
				str((req.TimeSeries.Ft)[t]),
				str((req.TimeSeries.Gt)[t]),
				str2((req.TimeSeries.PN)[t] * 1000.),
				str2((req.TimeSeries.PE)[t] * 1000.),
			}

			for _, bat := range req.Batteries {
				if bat.SGoal != nil && lo.Sum(*bat.SGoal) > 0 {
					row = append(row, str((*bat.SGoal)[t]))
				}
			}

			table.Append(row)
		}

		table.Render()
	}

	resp, err := c.PostOptimizeChargeScheduleWithResponse(context.TODO(), req)
	if err != nil {
		log.Fatal(err)
	}

	if resp.StatusCode() == http.StatusInternalServerError && resp.JSON500.Message != nil {
		log.Fatalf("Expected HTTP 200 but received %d\n%s", resp.StatusCode(), *resp.JSON500.Message)
	}

	if resp.StatusCode() != http.StatusOK {
		log.Fatalf("Expected HTTP 200 but received %d\n%s", resp.StatusCode(), string(resp.Body))
	}

	res := *resp.JSON200
	if *vFlag {
		b, _ := json.MarshalIndent(res, "", "  ")
		fmt.Println(string(b))
	}

	if *res.Status != "Optimal" {
		log.Fatal("Optimization failed:", string(*res.Status))
	}

	{
		table := tablewriter.NewTable(os.Stdout, tw)
		headers := []string{
			"Hour",
			// "Forecast",
			// "FlowDirection",
			"GridImport", "GridExport",
		}

		for i := range *res.Batteries {
			headers = append(headers,
				fmt.Sprintf("Bat %d Cha", i), // ChargingPower
				fmt.Sprintf("Bat %d Dis", i), // DischargingPower
				fmt.Sprintf("Bat %d Soc", i),
			)
		}

		table.Header(headers)

		for t := range len(*res.FlowDirection) {
			row := []string{
				strconv.Itoa(t + 1),
				// str((req.TimeSeries.Ft)[t]),
				// str((*res.FlowDirection)[t]),
				str((*res.GridImport)[t]),
				str((*res.GridExport)[t]),
			}

			for j, b := range *res.Batteries {
				_ = j
				row = append(row,
					str((*b.ChargingPower)[t]),
					str((*b.DischargingPower)[t]),
					str((*b.StateOfCharge)[t]),
					// str((*b.StateOfCharge)[i]/req.Batteries[j].SMax*100),
				)
			}

			table.Append(row)
		}

		table.Render()
	}

	{
		var power, soc [][]float64

		power = append(power, toFloat64Slice(*res.GridImport, 1))
		power = append(power, toFloat64Slice(*res.GridExport, 1))
		power = append(power, toFloat64Slice(req.TimeSeries.Ft, 1))

		powerSeries := []string{"Grid Import", "Grid Export", "Forecast"}
		var socSeries []string

		for i, b := range *res.Batteries {
			powerSeries = append(powerSeries,
				fmt.Sprintf("Bat %d Charge Power", i+1),
				fmt.Sprintf("Bat %d Discharge Power", i+1),
			)
			socSeries = append(socSeries, fmt.Sprintf("Bat %d SoC", i+1))

			power = append(power, toFloat64Slice(*b.ChargingPower, 1))
			power = append(power, toFloat64Slice(*b.DischargingPower, 1))
			soc = append(soc, toFloat64Slice(*b.StateOfCharge, req.Batteries[i].SMax/100))
		}

		fmt.Println(asciigraph.PlotMany(soc, asciigraph.Precision(1),
			asciigraph.Width(*cwFlag),
			asciigraph.Height(*chFlag/2),
			asciigraph.Caption("Optimization - SoC"),
			asciigraph.SeriesLegends(socSeries...),
			asciigraph.SeriesColors(lo.RepeatBy(len(socSeries), func(i int) asciigraph.AnsiColor {
				switch i % 3 {
				case 0:
					return asciigraph.Green
				case 1:
					return asciigraph.DarkOrange
				default:
					return asciigraph.Green
				}
			})...),
		))

		fmt.Println(asciigraph.PlotMany(power, asciigraph.Precision(0),
			asciigraph.Width(*cwFlag),
			asciigraph.Height(*chFlag),
			asciigraph.Caption("Optimization - Power Flow"),
			asciigraph.SeriesLegends(powerSeries...),
			asciigraph.SeriesColors(lo.RepeatBy(len(powerSeries), func(i int) asciigraph.AnsiColor {
				switch i {
				case 0:
					return asciigraph.LightBlue
				case 1:
					return asciigraph.Blue
				case 2:
					return asciigraph.Yellow
				case 3:
					return asciigraph.Green
				case 4:
					return asciigraph.DarkGreen
				case 5:
					return asciigraph.DarkOrange
				case 6:
					return asciigraph.DarkRed
				default:
					return asciigraph.White
				}
			})...),
		))
	}
}

func str(f float32) string {
	if f == 0 {
		return "-"
	}
	return fmt.Sprintf("%.0f", f)
}

func str2(f float32) string {
	if f == 0 {
		return "-"
	}
	return fmt.Sprintf("%.2f", f)
}

// toFloat64Slice converts a slice of float32 to a slice of float64.
func toFloat64Slice(in []float32, div float32) []float64 {
	out := make([]float64, len(in))
	for i, v := range in {
		out[i] = float64(v / div)
	}
	return out
}
